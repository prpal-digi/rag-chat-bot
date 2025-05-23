import requests
import re

API_BASE_URL = "http://localhost:8000/subscriptions"  # or your actual FastAPI base URL

def extract_user_id(text):
    match = re.search(r'user\s*id\s*(\d+)|user\s*(\d+)', text, re.IGNORECASE)
    if match:
        return match.group(1) or match.group(2)
    return "1"  # default fallback

def extract_plan_id(text):
    # You can customize this list with your actual plan names
    plan_keywords = ["basic", "standard", "premium", "pro", "gold", "silver", "diamond"]
    for plan in plan_keywords:
        if plan in text.lower():
            return plan
    return "basic"  # default fallback

def extract_subscription_id(text):
    match = re.search(r'subscription\s*id\s*(\d+)|subscription\s*(\d+)|id\s*(\d+)', text, re.IGNORECASE)
    if match:
        return match.group(1) or match.group(2) or match.group(3)
    return "1"  # default fallback

def fetch_api_data():
    try:
        response = requests.get(f"{API_BASE_URL}/all")
        response.raise_for_status()
        data = response.json()
        docs = []
        for item in data:
            text = f"Subscription ID: {item['id']}\nPlan: {item['plan_id']}\nStatus: {item['status']}"
            docs.append({"page_content": text, "metadata": {"source": "api"}})
        return docs
    except Exception as e:
        print(f"Error fetching API data: {e}")
        return []

def create_subscription(user_input):
    user_id = extract_user_id(user_input)
    plan_id = extract_plan_id(user_input)
    payload = {"user_id": user_id, "plan_id": plan_id}
    try:
        response = requests.post(API_BASE_URL, json=payload)
        return response.json()
    except Exception as e:
        return f"Failed to create subscription: {e}"

def update_subscription(user_input):
    subscription_id = extract_subscription_id(user_input)
    new_plan_id = extract_plan_id(user_input)
    try:
        response = requests.put(f"{API_BASE_URL}/{subscription_id}", json={"plan_id": new_plan_id})
        return response.json()
    except Exception as e:
        return f"Failed to update subscription: {e}"

def delete_subscription(user_input):
    subscription_id = extract_subscription_id(user_input)
    try:
        response = requests.delete(f"{API_BASE_URL}/{subscription_id}")
        return {"message": "Subscription deleted"} if response.ok else response.text
    except Exception as e:
        return f"Error deleting subscription: {e}"

def cancel_subscription(user_input):
    subscription_id = extract_subscription_id(user_input)
    try:
        response = requests.post(f"{API_BASE_URL}/{subscription_id}/cancel")
        return response.json()
    except Exception as e:
        return f"Error canceling subscription: {e}"

def get_subscription(user_input):
    subscription_id = extract_subscription_id(user_input)
    try:
        response = requests.get(f"{API_BASE_URL}/{subscription_id}")
        return response.json()
    except Exception as e:
        return f"Error getting subscription: {e}"
# --- Utility ---

def extract_id(text):
    import re
    match = re.search(r'\d+', text)
    return match.group() if match else "1"  # Default for testing

def intent_classifier(text):
    text = text.lower()
    if "create" in text:
        return "create"
    elif "get all" in text or "show all" in text:
        return "get_all"
    elif "get" in text or "fetch" in text:
        return "get"
    elif "update" in text or "change" in text:
        return "update"
    elif "delete" in text or "remove" in text:
        return "delete"
    elif "cancel" in text:
        return "cancel"
    else:
        return "doc_query"
