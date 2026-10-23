import boto3
import json


def generate_executive_report(var_normal, cvar_normal, var_stress, cvar_stress, corner_asset):
    print("\n🤖 Initiating AWS Bedrock AI Analysis...")

    # Initialize the AWS Bedrock client with direct credentials
    try:
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1',
            aws_access_key_id='',
            aws_secret_access_key=''
        )
    except Exception as e:
        print(f"❌ AWS Credentials not found or invalid: {e}")
        return

    prompt_text = f"""
    You are a Chief Risk Officer. Write an Executive Risk Memorandum based on the following stochastic stress test results for a Tadawul portfolio.
    Simulation Parameters: 10,000 Monte Carlo Pathways.
    Stress Test Allocation: The optimization algorithm converged to a 100% corner solution in {corner_asset}.

    Baseline Exposure: 
    - 95% Value at Risk (VaR): {var_normal:,.2f} SAR
    - Conditional Value at Risk (CVaR): {cvar_normal:,.2f} SAR

    Stress Exposure (Geopolitical Shock):
    - 95% Value at Risk (VaR): {var_stress:,.2f} SAR
    - Conditional Value at Risk (CVaR): {cvar_stress:,.2f} SAR

    Instructions:
    Write a 3-paragraph executive summary detailing the structural allocation shift, the escalation in tail risk (calculate the exact percentage increase in CVaR), and a brief strategic recommendation. Maintain a highly formal financial tone.
    """

    try:
        # Invoke Amazon Nova lite using the universal Converse API
        response = bedrock.converse(
            modelId='amazon.nova-lite-v1:0',
            messages=[{
                "role": "user",
                "content": [{"text": prompt_text}]
            }],
            inferenceConfig={
                "maxTokens": 600,
                "temperature": 0.1
            }
        )

        # Extract the text from the Converse response
        executive_report = response['output']['message']['content'][0]['text']

        print("=" * 80)
        print(" " * 20 + "📊 AWS AI-GENERATED RISK REPORT 📊")
        print("=" * 80)
        print(executive_report)
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error generating report from AWS Bedrock: {e}")
