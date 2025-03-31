import streamlit as st
import base64
import json
import io
from PIL import Image
from openai import OpenAI

# Initialize OpenAI client with your API key (hardcoded)
client = OpenAI(api_key="")

def convert_to_png_and_encode(file) -> str:
    """
    Convert the uploaded image to PNG (lossless) and then encode it to a Base64 string.
    """
    # Open the image with Pillow
    image = Image.open(file)
    # Create a BytesIO buffer to save the image as PNG
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    # Encode the PNG image to Base64
    return base64.b64encode(buffer.read()).decode("utf-8")

def main():
    st.title("Structured Text Extraction from Image")
    st.write(
        "Upload an image file. The image will be converted to PNG (lossless), then Base64 encoded, "
        "and sent to GPT to extract all text in a structured way if present in the image."
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp", "gif"])
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Convert the image to PNG and encode it in Base64.
        base64_image = convert_to_png_and_encode(uploaded_file)
        # Construct a data URI with MIME type image/png since we've converted it.
        data_uri = f"data:image/png;base64,{base64_image}"

        # Optionally display the Base64 encoded string
        st.text_area("Base64 Encoded Image", data_uri, height=200)

        if st.button("Extract Structured Text"):
            with st.spinner("Calling GPT..."):
                try:
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": """Analyze the given heatmap image and extract key insights from it. The heatmap represents the correlation matrix of a dataset, where each cell shows the correlation coefficient between two attributes. Follow these steps in your analysis:

Identify Strong Correlations: List attributes that have strong positive (above +0.7) and strong negative (below -0.7) correlations.

Detect Moderate Correlations: Highlight moderately correlated attributes (+0.4 to +0.7 or -0.4 to -0.7).

Find Independent Features: Identify features with low or no correlation (close to 0).

Interpret Key Relationships: Explain significant correlations and their possible impact (e.g., how one feature influences another).

Highlight Anomalies or Unexpected Correlations: Mention any surprising relationships in the data.

Summarize Findings: Provide a concise summary of the most important patterns observed in the heatmap.

Provide a structured response with clear bullet points and reasoning behind the observed correlations."""


                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": data_uri},
                                    },
                                ],
                            }
                        ],
                        temperature=0,  # For deterministic output
                    )
                    # Extract structured text from the GPT response
                    answer = completion.choices[0].message.content
                    st.subheader("Extracted Structured Text")
                    st.text_area("Structured Text", answer, height=300)

                    st.subheader("Full GPT Response")
                    # Attempt to serialize the entire response object using a default function
                    try:
                        full_response = json.dumps(completion, default=lambda o: o.__dict__, indent=2)
                    except Exception as ser_e:
                        full_response = str(completion)
                    st.text_area("GPT Response", full_response, height=300)

                    # Optionally display token usage if available
                    if hasattr(completion, "usage"):
                        st.write("Token Usage:", completion.usage)
                    elif isinstance(completion, dict) and "usage" in completion:
                        st.write("Token Usage:", completion["usage"])
                except Exception as e:
                    st.error(f"Error calling GPT: {e}")

if __name__ == '__main__':
    main()
