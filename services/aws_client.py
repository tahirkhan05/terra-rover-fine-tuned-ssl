import boto3
import time
import json
import base64
from botocore.exceptions import BotoCoreError, ClientError
from config.settings import settings
from utils.logger import logger

class AWSClient:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AWSClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.session = boto3.Session(
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.s3 = self.session.client('s3')
        # Initialize bedrock and lex clients
        self.bedrock = self.session.client('bedrock-runtime')
        self.lex = self.session.client('lexv2-runtime')
        
        self._initialized = True
        self.lifecycle_configured = False
        self._configure_bucket()
    
    def _configure_bucket(self):
        """Ensure bucket is properly configured"""
        try:
            self.s3.head_bucket(Bucket=settings.S3_BUCKET)
            if not self.lifecycle_configured:  # Only configure once
                self._setup_lifecycle()
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                self._create_bucket()
        
    def _create_bucket(self):
        """Create S3 bucket if it doesn't exist"""
        try:
            # US East-1 requires a different format than other regions
            if settings.AWS_REGION == 'us-east-1':
                self.s3.create_bucket(Bucket=settings.S3_BUCKET)
            else:
                self.s3.create_bucket(
                    Bucket=settings.S3_BUCKET,
                    CreateBucketConfiguration={
                        'LocationConstraint': settings.AWS_REGION
                    }
                )
            self._setup_lifecycle()
            logger.info(f"Created S3 bucket: {settings.S3_BUCKET}")
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Bucket creation failed: {str(e)}")
            raise

    def _setup_lifecycle(self):
        """Configure automatic deletion of old frames"""
        if self.lifecycle_configured:  # Skip if already configured
            return
            
        try:
            self.s3.put_bucket_lifecycle_configuration(
                Bucket=settings.S3_BUCKET,
                LifecycleConfiguration={
                    'Rules': [{
                        'ID': 'auto-delete-frames',
                        'Status': 'Enabled',
                        'Filter': {'Prefix': 'frames/'},
                        'Expiration': {'Days': 3}
                    }]
                }
            )
            self.lifecycle_configured = True
            logger.info("S3 lifecycle policy configured successfully")
        except Exception as e:
            logger.error(f"Lifecycle configuration failed: {str(e)}")
        
    def get_presigned_url(self, key, expiration=3600):
        try:
            return self.s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': settings.S3_BUCKET, 'Key': key},
                ExpiresIn=expiration
            )
        except (BotoCoreError, ClientError) as e:
            logger.error(f"S3 presigned URL error: {str(e)}")
            return None
    
    def invoke_vlm(self, prompt, image_key=None):
        try:
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }]

            if image_key:
                # Get image bytes from S3 and convert to base64
                response = self.s3.get_object(
                    Bucket=settings.S3_BUCKET,
                    Key=image_key
                )
                image_bytes = response['Body'].read()
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                
                # Add image to messages
                messages[0]["content"].insert(0, {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                })

            response = self.bedrock.invoke_model(
                modelId=settings.VLM_MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1024,
                    "messages": messages
                })
            )
            
            return response['body'].read().decode('utf-8')
        except (BotoCoreError, ClientError) as e:
            logger.error(f"VLM invocation error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in VLM invocation: {str(e)}")
            return None
    
    def recognize_speech(self, audio_bytes):
        """Process audio with Amazon Lex and return transcription"""
        if not audio_bytes:
            logger.error("No audio data provided")
            return None
            
        try:
            # Generate unique session ID
            session_id = f"terra-rover-{int(time.time())}"
            
            # Log request details for debugging
            logger.debug(f"Sending speech recognition request to Lex")
            logger.debug(f"Bot ID: {settings.LEX_BOT_ID}")
            logger.debug(f"Bot Alias ID: {settings.LEX_BOT_ALIAS_ID}")
            logger.debug(f"Locale ID: {settings.LEX_LOCALE_ID}")
            logger.debug(f"Session ID: {session_id}")
            
            # Make the Lex API call
            response = self.lex.recognize_utterance(
                botId=settings.LEX_BOT_ID,
                botAliasId=settings.LEX_BOT_ALIAS_ID,
                localeId=settings.LEX_LOCALE_ID,
                sessionId=session_id,
                requestContentType='audio/l16; rate=16000; channels=1',
                responseContentType='text/plain; charset=utf-8',
                inputStream=audio_bytes
            )
            
            # Debug response
            logger.debug(f"Raw Lex response: {response}")
            
            # Check for transcription in response
            transcription = None
            if 'inputTranscript' in response:
                transcription = response['inputTranscript']
                logger.info(f"Transcription: {transcription}")
            elif 'messages' in response and response['messages']:
                # Extract message content if available
                transcription = response['messages']
                logger.info(f"Lex message: {transcription}")
            else:
                logger.warning("No transcription or message in Lex response")
            
            return transcription
                
        except Exception as e:
            logger.error(f"Speech recognition error: {str(e)}")
            return None
