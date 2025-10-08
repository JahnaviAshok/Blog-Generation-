# lambda_function.py
import os
import json
from datetime import datetime
from base64 import b64decode

import boto3
import botocore.config

# ---- Config ----
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
# Adjust this to the exact model ID you have access to in Bedrock:
# e.g., "meta.llama3-8b-instruct-v1:0"
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.meta.llama3-1-8b-instruct-v1:0")

S3_BUCKET = os.environ.get("BLOG_S3_BUCKET", "bedrock-demo-blogs")

_bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    config=botocore.config.Config(read_timeout=200, retries={"max_attempts": 3})
)


def _ok(body: dict, status: int = 200, headers: dict | None = None):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json", **(headers or {})},
        "body": json.dumps(body),
    }


def _error(message: str, status: int = 500):
    return _ok({"error": message}, status=status)


def generate_blog(topic: str) -> str:
    """
    Calls Amazon Bedrock (Meta Llama 3 Instruct) to generate ~250 words.
    """
    prompt = f"<s>[INST]Human: Write a ~250-word blog on the topic: {topic}. Assistant:[/INST]"

    body = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9,
    }

    resp = _bedrock.invoke_model(
        body=json.dumps(body),
        modelId=BEDROCK_MODEL_ID,
        accept="application/json",
        contentType="application/json",
    )

    # Bedrock returns a StreamingBody under 'body'
    raw = resp["body"].read()
    data = json.loads(raw)

    # For Meta Llama on Bedrock, the text is typically under 'generation'
    text = data.get("generation") or ""
    return text.strip()


def save_blog_to_s3(key: str, content: str):
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=content.encode("utf-8"),
        ContentType="text/plain; charset=utf-8",
    )


def _parse_event_body(event: dict) -> dict:
    """
    Handles API Gateway proxy (string body) and base64 body cases.
    """
    if "body" not in event or event["body"] is None:
        return {}

    body_str = event["body"]
    if event.get("isBase64Encoded"):
        body_str = b64decode(body_str).decode("utf-8", errors="replace")
    try:
        return json.loads(body_str)
    except json.JSONDecodeError:
        return {}


def lambda_handler(event, context):
    # Expecting JSON like: {"blogstopic": "Your topic here"}
    body = _parse_event_body(event)
    topic = (body.get("blogstopic") or body.get("topic") or "").strip()

    if not topic:
        return _error("Missing required field 'blogstopic' in request body.", 400)

    try:
        blog_text = generate_blog(topic)
    except Exception as e:
        # Surface the error message for easier debugging
        return _error(f"Error generating blog: {e}", 502)

    if not blog_text:
        return _error("Model returned empty content.", 502)

    # Create a safe S3 key
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    safe_topic = "".join(c for c in topic if c.isalnum() or c in ("-", "_")).strip() or "blog"
    s3_key = f"blogs/{safe_topic}_{ts}.txt"

    try:
        save_blog_to_s3(s3_key, blog_text)
    except Exception as e:
        return _error(f"Failed to save to S3: {e}", 502)

    return _ok({
        "message": "Blog generated and saved.",
        "s3_bucket": S3_BUCKET,
        "s3_key": s3_key,
        "length_chars": len(blog_text),
        "preview": blog_text[:200] + ("..." if len(blog_text) > 200 else "")
    })
