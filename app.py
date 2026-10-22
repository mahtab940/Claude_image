import os
import base64
import uuid

import requests
import fal_client

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from anthropic import Anthropic, AnthropicError

load_dotenv()

app = Flask(__name__)

CLAUDE_MODEL = os.getenv(
    "CLAUDE_MODEL",
    "claude-sonnet-5-5"
)

IMAGE_MODEL = "fal-ai/flux/schnell"

SIZES = [
    "square",
    "square_hd",
    "landscape_4_3",
    "landscape_16_9",
    "portrait_4_3",
    "portrait_16_9"
]


@app.route("/")
def index():
    return render_template("index.html")


def improve_prompt(user_prompt):
    """Use Claude to improve the image prompt."""

    client = Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        timeout=60.0,
        max_retries=1
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=350,
        system=(
            "You are a professional AI image prompt engineer. "
            "Rewrite the user's description as a detailed "
            "image-generation prompt. Improve composition, "
            "lighting, colors, camera perspective and visual "
            "details when appropriate. Preserve the user's "
            "original subject, intentions and requested style. "
            "Do not add unrelated objects. "
            "Output only the improved prompt."
        ),
        messages=[
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    text = " ".join(
        block.text
        for block in response.content
        if block.type == "text"
    ).strip()

    if not text:
        raise ValueError("Claude returned an empty prompt.")

    return text


def generate_images(prompt, size, number):
    """Generate images with FLUX Schnell."""

    result = fal_client.subscribe(
        IMAGE_MODEL,
        arguments={
            "prompt": prompt,
            "image_size": size,
            "num_images": number,
            "num_inference_steps": 4,
            "output_format": "png",
            "enable_safety_checker": True
        }
    )

    images = result.get("images", [])

    if not images:
        raise ValueError("FLUX returned no images.")

    return images


def image_to_base64(url):
    """Fetch a generated image for preview and download."""

    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("Invalid image URL from provider.")

    response = requests.get(
        url,
        timeout=45,
        stream=True
    )

    try:
        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        if not content_type.startswith("image/png"):
            raise ValueError("Provider did not return a PNG image.")

        parts = []
        total = 0
        max_bytes = 20 * 1024 * 1024

        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                total += len(chunk)

                if total > max_bytes:
                    raise ValueError("Generated image is too large.")

                parts.append(chunk)

        image_bytes = b"".join(parts)

        if not image_bytes:
            raise ValueError("Generated image was empty.")

        return base64.b64encode(image_bytes).decode("utf-8")

    finally:
        response.close()


@app.route("/generate", methods=["POST"])
def generate():

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Invalid request format."
        }), 400

    prompt = data.get("prompt", "")
    size = data.get("size", "square")
    number = data.get("number", 1)
    use_claude = data.get("use_claude", True)

    if not isinstance(prompt, str):
        return jsonify({
            "error": "Prompt must be text."
        }), 400

    prompt = prompt.strip()

    if not prompt:
        return jsonify({
            "error": "Please enter an image prompt."
        }), 400

    if len(prompt) > 2000:
        return jsonify({
            "error": "Prompt is too long. Maximum 2000 characters."
        }), 400

    if size not in SIZES:
        return jsonify({
            "error": "Invalid image size."
        }), 400

    if type(number) is not int or number not in [1, 2, 3, 4]:
        return jsonify({
            "error": "Choose between 1 and 4 images."
        }), 400

    if type(use_claude) is not bool:
        return jsonify({
            "error": "Invalid Claude option."
        }), 400

    if not os.getenv("FAL_KEY"):
        return jsonify({
            "error": "FAL_KEY is missing from your .env file."
        }), 503

    if use_claude and not os.getenv("ANTHROPIC_API_KEY"):
        return jsonify({
            "error": "ANTHROPIC_API_KEY is missing."
        }), 503

    try:
        final_prompt = prompt

        if use_claude:
            final_prompt = improve_prompt(prompt)

        generated = generate_images(
            final_prompt,
            size,
            number
        )

        output = []

        for image in generated:
            url = image.get("url")

            encoded = image_to_base64(url)

            output.append({
                "image": encoded,
                "filename": (
                    f"ai-image-{uuid.uuid4().hex[:8]}.png"
                )
            })

        return jsonify({
            "success": True,
            "original_prompt": prompt,
            "improved_prompt": final_prompt,
            "claude_used": use_claude,
            "claude_model": CLAUDE_MODEL if use_claude else None,
            "image_model": IMAGE_MODEL,
            "size": size,
            "images": output
        })

    except AnthropicError as error:
        app.logger.warning("Claude API error: %s", error)

        return jsonify({
            "error": (
                "Claude API failed. Check your Anthropic "
                "API key, credits, model access and network."
            )
        }), 502

    except requests.RequestException:
        app.logger.exception("Image download error")

        return jsonify({
            "error": "Unable to retrieve generated image."
        }), 502

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 502

    except Exception:
        app.logger.exception("Image generation failed")

        return jsonify({
            "error": (
                "Image generation failed. Check your fal.ai "
                "API key, credits, model access and connection."
            )
        }), 502


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
