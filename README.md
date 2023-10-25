# LLM powered apps for tabular data
This repo will serve as a collection of LLM powered streamlit apps that allows you to chat with your CSV files, generate insights, create visualizations and more.

**Note that these are in development and further changes can be expected soon**

## Requirments
Generate your OpenAI API key here: [Click Here](https://platform.openai.com/account/api-keys)
```
pip install -r requirements.txt
```

## Run locally
Create a `.env` file in this folder and add the OpenAI API Key in it like below:
```
OPENAI_API_KEY=s#-#####################jz
```
This will be used in code using the below code:
```
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
```

**OR**

Unsafe but easy option for local testing, provide key directly in code:
```
openai_api_key = 's#-#####################jz'
```

**Run the apps by using below commands**:

```
streamlit run csvchatapp.py
streamlit run misinsightapp.py
```

## App Details

### csvchatapp.py
Simple app to chat with your CSV data using pandasAI agents which can execute code to get you answers

### mixinsightapp.py
Generates AI powered marketing mix based insights based on the data you provide in the CSV file and also generates simple plots in Streamlit to visualize the data.