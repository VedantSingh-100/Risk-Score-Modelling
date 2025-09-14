import pandas as pd
from openai import OpenAI

# Initialize client
client = OpenAI(api_key="sk-proj-6QZuV_N8mKb1aNQZpJG6E2g-7m8lU_O169V1K0a8hU828428603827082307")

# Load Excel file
file_path = "/home/vhsingh/Parshvi_project/Internal_Algo360VariableDictionary_WithExplanation.xlsx"  # <-- replace with your file path
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Ensure correct columns exist
if not {"New_Description", "Explanation"}.issubset(df.columns):
    raise ValueError("Excel must contain 'New_Description' and 'Explanation' columns in Sheet1")

# Define prompts
SYSTEM_PROMPT = """You are an expert risk model analyst. 
You will analyze variable descriptions and provide clear, comprehensive insights.
Always respond with structured, professional explanations suitable for risk model documentation."""

USER_PROMPT_TEMPLATE = """Variable Description: {desc}
Explanation: {expl}
---
Please expand this into a detailed and professional description, highlighting:
1. What this variable measures or represents
2. Why it could be important for risk modeling
3. Any potential limitations, caveats, or edge cases
4. Examples where this variable may signal risk or stability
"""

# Store responses
responses = []

for idx, row in df.iterrows():
    desc = str(row["New_Description"])
    expl = str(row["Explanation"])

    user_prompt = USER_PROMPT_TEMPLATE.format(desc=desc, expl=expl)

    # Call OpenAI GPT-4.1 API
    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=600
    )

    response_text = completion.choices[0].message.content
    responses.append(response_text)

# Add responses to DataFrame
df["GPT_Response"] = responses

# Save back to Excel
output_file = "processed_with_gpt.xlsx"
df.to_excel(output_file, index=False)

print(f"✅ Processing complete. Results saved to {output_file}")