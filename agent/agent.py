from openai import AsyncOpenAI
from agents import set_default_openai_client
from agents import Agent, Runner
from agents import Agent, OpenAIChatCompletionsModel
from agents import Agent, FunctionTool, RunContextWrapper, function_tool
import datarobot as dr 
import os
from datarobot_predict.deployment import predict
from dotenv import load_dotenv 
import json
import requests
import os
from datetime import datetime
from rapidfuzz import process
load_dotenv(override = True)
import datarobot as dr
import monitoring

try:
    dr_client = dr.Client()
    monitoring.configure_tracer(dr_client.token)
except Exception as e:
    print("tracing not available")

client = AsyncOpenAI(
    base_url="https://app.datarobot.com/api/v2/deployments/682cb3448a869c36cea3a77f",
    api_key=os.environ["DATAROBOT_API_TOKEN"],
)

local_stuff = {
    "data": None, 
    "predictions": None, 
    "email": None, 
}


@function_tool  
async def retrieve_dataset() -> str:
    """use this function when if a user needs data from datarobot, or has any questions about data, you should use this tool
    """
    client = dr.Client() 
    dataset = dr.Dataset.get("685964f13e74a8ab40b9ce06")
    df = dataset.get_as_dataframe()
    # In real life, we'd fetch the weather from a weather API
    local_stuff["data"] = df
    return "dataset has been grabbed and is available"

@function_tool  
async def make_all_predictions() -> str:
    
    """this function should be used when you need to make churn predictions on an entire dataset.
    """
    client = dr.Client() 
    try:
        df = local_stuff["data"]
        predictions = predict(dr.Deployment.get("6859d7f8e71e9e4c797238e4"), df, max_explanations = 3, passthrough_columns=["ACCOUNT_NAME"])
        local_stuff["predictions"] = predictions
    except Exception as e:
        return "data has not been retrieved, you should grab the churn data first before you try to make predictions"
    # In real life, we'd fetch the weather from a weather API
    return "predictions have been generated and are available for futher analysis"

@function_tool  
async def make_prediction(company: str) -> str:
    
    """this function should be used when you need to make churn predictions on a particular company.  The function will return prediction plus prediction explanations (i.e., drivers of the prediction)
    """
    client = dr.Client() 
    try:
        df = local_stuff["data"]
        try:
            df = df[df["ACCOUNT_NAME"] == company]
        except Exception as e:
            return "you probably need to sanitze the company name before you attempt to make any predictions"
        predictions = predict(dr.Deployment.get("6859d7f8e71e9e4c797238e4"), df, max_explanations = 3, passthrough_columns=["ACCOUNT_NAME"])
        local_stuff[f"{company}_predictions"] = predictions
    except Exception as e:
        return "data has not been retrieved, you should grab the churn data first before you try to make predictions"
    # In real life, we'd fetch the weather from a weather API
    return predictions.to_csv(index = False)

@function_tool
async def top_churn(k: int = 3):
    """
    this function should be used to return the customers with top chances of top churn 

    Args:
        k: the number of top k customers to return 
    """
    try:
        top_k_pred = local_stuff["predictions"].dataframe.sort_values("STAGE_NAME_8-Closed Lost_PREDICTION", ascending = False).head(k)
        local_stuff["top_k_preds"] = top_k_pred
        return top_k_pred.to_csv(index = False)
    except Exception as e:
        print(e)
        return "top k churn was not identifed.  Either you don't have data available, or you haven't made predictions"


@function_tool
async def generate_email(company: str):
    """
    use this function to generate an candidate email to send to a customer.  

    Args:
        company: the customer you want to send the email to. 
    """
    try:
        search_result = local_stuff["searches"][company]
    except Exception as e:
        print(e)
        return f"you did not search out any insights or other detail on {company}"
    try:
        predictions = local_stuff[f"{company}_predictions"]
    except Exception as e:
        print(e)
        return f"you did not complete churn predictions for {company}"

    prompt = f"""
    Generate an email to generate excitement and interest from a technical executive like CTO, CIO, director of data science and analytics. If provided information or direction about new AI initiatives, make sure to mention them as a reason you are emailing them.
    Mention DataRobot has new exciting capabilities to empower teams and organizations in the future of Agentic AI and ask for time to meet to discuss.

    Here is information regarding an internet search of {company}: {search_result}

    Here is also information regarding churn prediction for {company}: \n{predictions.dataframe.to_csv(index=False)}
    """

    email = await client.chat.completions.create(model="gpt-4o",
                                      messages=[{"role": "user", "content": prompt}]
    )
    local_stuff[f"{company}_email"] = email.choices[0].message.content
    return email.choices[0].message.content


@function_tool
async def sanitize_company_name(company: str): 
    """
    this function should be to sanitize the name of the company that you need to search the web, or if you need to request a churn prediction

    Args:
        company: the company that you need to search for more information on. 
    """   
    try:
        accounts = local_stuff["data"]["ACCOUNT_NAME"].values.tolist()
        best_match = process.extractOne(company, accounts)
        return best_match[0]
    except Exception as e:
        print(e)
        return "Chances are you did not retrieve the data yet"

@function_tool
async def perplexity_search(company: str):
    """
    this function should be used when searching for additional information about customers that are likely to churn

    Args:
        company: the company that you need to search for more information on. 
    """
    churn_accounts = local_stuff["top_k_preds"]["ACCOUNT_NAME"].values.tolist()
    searches = local_stuff.get("searches", {})
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system","content": "Be precise and concise."},
            {"role": "user","content": f"Search for news regarding ai initiatives or trends that might apply to {company}"}  
        ]
    }
    headers = {
        "Authorization": f"Bearer {os.environ.get('PERPLEXITY_API_KEY')}",
        "Content-Type": "application/json"
    }
    search_response = requests.request("POST", url, json=payload, headers=headers)
    searches[company] = search_response.json()["choices"][0]["message"]["content"]
    local_stuff["searches"] = searches
    return searches[company]


model=OpenAIChatCompletionsModel(model="gpt-4o", openai_client=client)    

agent = Agent(name="Assistant", instructions="You are a helpful assistant", model=model, tools=[retrieve_dataset, make_all_predictions, make_prediction, perplexity_search, top_churn, sanitize_company_name, generate_email])

