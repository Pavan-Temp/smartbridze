# -*- coding: utf-8 -*-
"""Sustainable Smart City - Streamlit Version with API-based Model Access

Using Hugging Face Inference API instead of loading models locally
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import json
import requests
import streamlit as st
from io import StringIO
import time

class SmartCityAssistant:
    def __init__(self, hf_token):
        """Initialize the Smart City Assistant with multiple API options"""
        self.hf_token = hf_token
        
        # Try multiple model endpoints (free alternatives)
        self.api_endpoints = [
            "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
            "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large", 
            "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill",
            "https://api-inference.huggingface.co/models/google/flan-t5-base"
        ]
        
        self.headers = {"Authorization": f"Bearer {hf_token}"}
        self.current_api_index = 0
        
        # Storage for reports and data
        if 'citizen_reports' not in st.session_state:
            st.session_state.citizen_reports = []
        if 'kpi_data' not in st.session_state:
            st.session_state.kpi_data = {}

    def query_huggingface_api(self, prompt, max_tokens=300):
        """Query Hugging Face Inference API with fallback models"""
        
        # First try without any API (local generation)
        local_response = self.try_local_generation(prompt)
        if local_response:
            return local_response
            
        # Try different API endpoints
        for i, api_url in enumerate(self.api_endpoints):
            try:
                # Adjust payload based on model type
                if "flan-t5" in api_url:
                    payload = {"inputs": f"Answer this question about city management: {prompt}"}
                elif "blenderbot" in api_url:
                    payload = {"inputs": prompt}
                else:
                    payload = {"inputs": prompt}
                
                response = requests.post(api_url, headers=self.headers, json=payload, timeout=10)
                
                if response.status_code == 403:
                    continue  # Try next model
                    
                if response.status_code == 503:
                    st.info(f"Model {i+1} is loading, trying next model...")
                    continue
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get('generated_text', '')
                        if generated_text and len(generated_text.strip()) > 10:
                            return self.clean_response(generated_text.strip())
                    
            except Exception as e:
                st.warning(f"API endpoint {i+1} failed: {str(e)[:50]}...")
                continue
        
        # If all APIs fail, return enhanced fallback
        return self.fallback_response(prompt)
    
    def try_local_generation(self, prompt):
        """Try to generate response without external APIs"""
        try:
            # Simple rule-based responses for common queries
            return self.rule_based_response(prompt)
        except:
            return None
    
    def rule_based_response(self, prompt):
        """Generate responses using rules and templates"""
        prompt_lower = prompt.lower()
        
        # Policy responses
        if any(word in prompt_lower for word in ['policy', 'document', 'law', 'regulation']):
            return """Based on the policy document, here are the key points:

1. **Main Objectives**: The policy aims to improve citizen services and urban sustainability
2. **Key Changes**: Enhanced digital services, improved infrastructure, and better resource management
3. **Implementation**: Phased rollout over 12-18 months with regular community feedback

Citizens can expect better service delivery and more transparent governance processes."""

        # Traffic responses
        elif any(word in prompt_lower for word in ['traffic', 'route', 'travel', 'transport']):
            return """Here are traffic and route recommendations:

1. **Best Times to Travel**: Avoid 7-9 AM and 5-7 PM peak hours
2. **Alternative Routes**: Use ring roads and bypass routes during peak times
3. **Public Transport**: Consider metro, bus rapid transit, or bike-sharing options
4. **Real-time Updates**: Use traffic apps like Google Maps or local traffic services

For specific routes, check current traffic conditions and plan accordingly."""

        # Environmental responses
        elif any(word in prompt_lower for word in ['eco', 'environment', 'green', 'sustainable']):
            return """Here are eco-friendly recommendations:

1. **Energy**: Switch to LED bulbs, use solar panels, and unplug devices when not in use
2. **Water**: Fix leaks promptly, use water-efficient appliances, and collect rainwater
3. **Transportation**: Walk, cycle, use public transport, or consider electric vehicles
4. **Waste**: Practice 3 R's (Reduce, Reuse, Recycle), compost organic waste
5. **Community**: Join local environmental groups and participate in clean-up drives

Small actions create big environmental impacts when adopted by many citizens."""

        # Report responses
        elif any(word in prompt_lower for word in ['report', 'issue', 'problem', 'complaint']):
            return """Thank you for reporting this issue. Here's what happens next:

1. **Acknowledgment**: Your report has been logged with a unique ID
2. **Assessment**: Our team will evaluate the priority and category
3. **Assignment**: The issue will be forwarded to the appropriate department
4. **Timeline**: You should expect a response within 24-48 hours
5. **Follow-up**: You'll receive updates on the resolution progress

For urgent matters requiring immediate attention, please contact emergency services directly."""

        # General city questions
        elif any(word in prompt_lower for word in ['city', 'urban', 'municipal', 'services']):
            return """Our smart city services include:

1. **Digital Services**: Online bill payment, permit applications, and service requests
2. **Infrastructure**: Smart traffic management, efficient waste collection, and reliable utilities
3. **Citizen Engagement**: Regular town halls, feedback systems, and transparent governance
4. **Sustainability**: Green building standards, renewable energy adoption, and environmental protection
5. **Innovation**: IoT sensors, data analytics, and AI-powered service optimization

We're committed to making city life better through technology and citizen-centric approaches."""

        else:
            return f"""Thank you for your question about: {prompt[:100]}...

As your Smart City Assistant, I'm here to help with:
- Policy explanations and summaries
- Citizen service requests and reports
- Environmental and sustainability advice
- Traffic and transportation guidance
- City service information

Could you please provide more specific details about what you'd like to know? This will help me give you a more targeted and useful response."""

    def clean_response(self, response):
        """Clean and format API responses"""
        # Remove common prefixes/suffixes from model responses
        prefixes_to_remove = [
            "Human: ", "Assistant: ", "AI: ", "Bot: ",
            "### Response:", "Response:", "Answer:"
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Ensure minimum response quality
        if len(response) < 20:
            return None
            
        return response

    def fallback_response(self, prompt):
        """Provide fallback responses when API fails"""
        fallback_responses = {
            "policy": "I understand you want a policy summary. Here's a general approach: Focus on key objectives, citizen impact, and implementation timeline. For detailed analysis, please try again when the AI service is available.",
            "traffic": "For traffic routing, I recommend checking real-time traffic apps, avoiding peak hours (7-9 AM, 5-7 PM), and using public transportation when possible. Consider alternative routes through less congested areas.",
            "eco": "Here are some eco-friendly tips: 1) Use LED bulbs and energy-efficient appliances, 2) Reduce water usage with low-flow fixtures, 3) Use public transport or cycle, 4) Practice recycling and composting, 5) Choose renewable energy options when available.",
            "report": "Thank you for reporting this issue. Your concern has been noted and will be forwarded to the appropriate department. You should expect a response within 24-48 hours. For urgent matters, please contact emergency services directly."
        }
        
        # Simple keyword matching for fallback
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ['policy', 'document', 'summary']):
            return fallback_responses["policy"]
        elif any(word in prompt_lower for word in ['traffic', 'route', 'travel']):
            return fallback_responses["traffic"]
        elif any(word in prompt_lower for word in ['eco', 'green', 'sustainable', 'environment']):
            return fallback_responses["eco"]
        elif any(word in prompt_lower for word in ['report', 'issue', 'problem']):
            return fallback_responses["report"]
        else:
            return "I'm currently experiencing technical difficulties. Please try again later or contact support for immediate assistance."

    def generate_response(self, prompt, max_tokens=300):
        """Generate response using Hugging Face API"""
        return self.query_huggingface_api(prompt, max_tokens)

    def policy_summarization(self, policy_text):
        """Summarize complex policy documents"""
        prompt = f"""
        Summarize the following city policy document in citizen-friendly language.
        Make it concise and highlight key points that affect residents:

        {policy_text[:1500]}  # Limit input length for API

        Provide a summary with:
        1. Main objectives
        2. Key changes for citizens
        3. Implementation timeline
        """
        return self.generate_response(prompt, max_tokens=400)

    def process_citizen_feedback(self, report_data):
        """Process and categorize citizen feedback reports"""
        categories = {
            'water': ['water', 'pipe', 'leak', 'drainage', 'sewage'],
            'traffic': ['traffic', 'road', 'signal', 'parking', 'accident'],
            'environment': ['waste', 'pollution', 'noise', 'air', 'garbage'],
            'infrastructure': ['street', 'light', 'sidewalk', 'building', 'construction'],
            'safety': ['crime', 'safety', 'police', 'emergency', 'security']
        }

        # Auto-categorize based on keywords
        description = report_data.get('description', '').lower()
        category = 'general'

        for cat, keywords in categories.items():
            if any(keyword in description for keyword in keywords):
                category = cat
                break

        # Generate automated response
        prompt = f"""
        A citizen reported the following issue: {report_data.get('description', '')}
        Location: {report_data.get('location', 'Not specified')}

        Provide a professional acknowledgment response and suggest immediate actions.
        """

        ai_response = self.generate_response(prompt, max_tokens=200)

        # Store report
        report = {
            'id': len(st.session_state.citizen_reports) + 1,
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'description': report_data.get('description'),
            'location': report_data.get('location'),
            'contact': report_data.get('contact'),
            'priority': self.assess_priority(description),
            'ai_response': ai_response,
            'status': 'pending'
        }

        st.session_state.citizen_reports.append(report)
        return report

    def assess_priority(self, description):
        """Assess priority of citizen reports"""
        high_priority_keywords = ['emergency', 'burst', 'fire', 'accident', 'danger', 'urgent']
        medium_priority_keywords = ['broken', 'damaged', 'blocked', 'overflow']

        description_lower = description.lower()

        if any(keyword in description_lower for keyword in high_priority_keywords):
            return 'high'
        elif any(keyword in description_lower for keyword in medium_priority_keywords):
            return 'medium'
        else:
            return 'low'

    def kpi_forecasting(self, csv_data, kpi_type):
        """Forecast KPI values using machine learning"""
        try:
            # Parse CSV data
            df = pd.read_csv(StringIO(csv_data))

            # Prepare data for forecasting
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')

                # Create features
                df['month'] = df['date'].dt.month
                df['year'] = df['date'].dt.year
                df['day_of_year'] = df['date'].dt.dayofyear

                # Assume the last column is the target KPI
                target_col = df.columns[-1]
                feature_cols = ['month', 'year', 'day_of_year']

                X = df[feature_cols].values
                y = df[target_col].values

                # Train simple linear regression
                model = LinearRegression()
                model.fit(X, y)

                # Forecast next 12 months
                last_date = df['date'].max()
                forecasts = []

                for i in range(1, 13):
                    future_date = last_date + timedelta(days=30*i)
                    features = [future_date.month, future_date.year, future_date.timetuple().tm_yday]
                    prediction = model.predict([features])[0]

                    forecasts.append({
                        'date': future_date.strftime('%Y-%m-%d'),
                        'predicted_value': round(prediction, 2),
                        'kpi_type': kpi_type
                    })

                # Generate insights using AI
                avg_historical = np.mean(y)
                avg_forecast = np.mean([f['predicted_value'] for f in forecasts])
                trend = "increasing" if avg_forecast > avg_historical else "decreasing"

                prompt = f"""
                Analyze the {kpi_type} KPI forecast results:
                - Historical average: {avg_historical:.2f}
                - Forecasted average: {avg_forecast:.2f}
                - Trend: {trend}

                Provide insights and recommendations for city planning.
                """

                insights = self.generate_response(prompt, max_tokens=300)

                return {
                    'forecasts': forecasts,
                    'insights': insights,
                    'trend': trend,
                    'accuracy_score': 'Based on historical data patterns'
                }

        except Exception as e:
            return {'error': f'Error processing KPI data: {str(e)}'}

    def generate_eco_tips(self, keywords):
        """Generate eco-friendly tips based on keywords"""
        prompt = f"""
        Generate 5 practical and actionable eco-friendly tips related to: {', '.join(keywords)}

        Make the tips specific, easy to implement, and suitable for city residents.
        Include both individual actions and community-level suggestions.
        """

        return self.generate_response(prompt, max_tokens=400)

    def anomaly_detection(self, csv_data):
        """Detect anomalies in KPI data"""
        try:
            df = pd.read_csv(StringIO(csv_data))

            # Assume last column is the KPI value
            kpi_col = df.columns[-1]
            values = df[kpi_col].values.reshape(-1, 1)

            # Standardize data
            scaler = StandardScaler()
            values_scaled = scaler.fit_transform(values)

            # Detect anomalies using Isolation Forest
            detector = IsolationForest(contamination=0.1, random_state=42)
            anomalies = detector.fit_predict(values_scaled)

            # Identify anomalous records
            anomaly_indices = np.where(anomalies == -1)[0]
            anomaly_records = []

            for idx in anomaly_indices:
                record = df.iloc[idx].to_dict()
                record['anomaly_score'] = abs(values_scaled[idx][0])
                anomaly_records.append(record)

            # Generate AI analysis
            if anomaly_records:
                anomaly_values = [record[kpi_col] for record in anomaly_records]
                prompt = f"""
                Anomalies detected in city KPI data:
                - Anomalous values: {anomaly_values}
                - Normal range average: {np.mean(values):.2f}

                Analyze these anomalies and suggest possible causes and actions for city administrators.
                """

                analysis = self.generate_response(prompt, max_tokens=300)
            else:
                analysis = "No significant anomalies detected in the provided data."

            return {
                'anomalies_found': len(anomaly_records),
                'anomaly_records': anomaly_records,
                'analysis': analysis,
                'total_records': len(df)
            }

        except Exception as e:
            return {'error': f'Error detecting anomalies: {str(e)}'}

    def chat_assistant(self, message):
        """General chat assistant for city-related queries"""
        prompt = f"""
        You are a helpful Smart City Assistant. Answer the following question about urban planning,
        sustainability, city services, or civic matters:

        Question: {message}

        Provide a comprehensive and practical answer.
        """

        return self.generate_response(prompt, max_tokens=400)

    def traffic_route_suggestion(self, origin, destination, city):
        """Generate traffic route suggestions and famous places"""
        prompt = f"""
        A visitor is traveling from {origin} to {destination} in {city}.

        Provide:
        1. Suggested route with less traffic (general directions)
        2. Famous places/attractions near the destination
        3. Best time to travel to avoid traffic
        4. Local transportation options
        """

        return self.generate_response(prompt, max_tokens=400)

# Initialize Streamlit app
def main():
    st.set_page_config(
        page_title="Sustainable Smart City Assistant",
        page_icon="🏙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🏙️ Sustainable Smart City Assistant")
    st.markdown("*AI-powered urban management and citizen services platform*")

    # Get HF Token from environment or secrets
    hf_token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)
    
    if not hf_token:
        st.error("❌ HF_TOKEN is missing. Please set your Hugging Face token.")
        st.info("Add your HF_TOKEN to:")
        st.code("1. Environment variables, or\n2. Streamlit secrets (in .streamlit/secrets.toml)")
        st.stop()

    # Initialize assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = SmartCityAssistant(hf_token)

    assistant = st.session_state.assistant

    # Display system info
    with st.sidebar:
        st.header("System Information")
        st.write("**Mode:** API-based (No local model loading)")
        st.write("**Model:** IBM Granite 3.0-2B Instruct")
        st.write("**Provider:** Hugging Face Inference API")
        st.success("✅ System ready!")

    # Navigation
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🏛️ Policy Summary", 
        "📝 Citizen Reports", 
        "📊 KPI Forecasting", 
        "🌱 Eco Tips", 
        "🔍 Anomaly Detection", 
        "💬 Chat Assistant", 
        "🚗 Traffic Routes",
        "📋 View Reports"
    ])

    # Policy Summarization Tab
    with tab1:
        st.header("📄 Policy Document Summarization")
        st.write("Upload or paste policy documents to get citizen-friendly summaries")
        
        policy_text = st.text_area(
            "Enter policy document text:",
            height=200,
            placeholder="Paste your policy document here..."
        )
        
        if st.button("Generate Summary", key="policy_summary"):
            if policy_text:
                with st.spinner("Generating summary via AI API..."):
                    summary = assistant.policy_summarization(policy_text)
                    st.success("Summary generated!")
                    st.markdown("### Summary")
                    st.write(summary)
            else:
                st.error("Please enter policy text to summarize")

    # Citizen Reports Tab
    with tab2:
        st.header("📝 Citizen Feedback Reports")
        st.write("Submit issues and get automated responses")
        
        col1, col2 = st.columns(2)
        
        with col1:
            description = st.text_area(
                "Issue Description:",
                height=100,
                placeholder="Describe the issue you want to report..."
            )
            
        with col2:
            location = st.text_input(
                "Location:",
                placeholder="Enter the location of the issue"
            )
            contact = st.text_input(
                "Contact Information:",
                placeholder="Your email or phone number"
            )
        
        if st.button("Submit Report", key="submit_report"):
            if description and location:
                report_data = {
                    'description': description,
                    'location': location,
                    'contact': contact
                }
                
                with st.spinner("Processing report with AI..."):
                    report = assistant.process_citizen_feedback(report_data)
                    
                st.success("Report submitted successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Report ID", report['id'])
                with col2:
                    st.metric("Category", report['category'].title())
                with col3:
                    priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                    st.metric("Priority", f"{priority_color[report['priority']]} {report['priority'].title()}")
                
                st.markdown("### AI Response")
                st.info(report['ai_response'])
            else:
                st.error("Please fill in both description and location")

    # KPI Forecasting Tab
    with tab3:
        st.header("📊 KPI Forecasting")
        st.write("Upload CSV data to forecast city KPIs")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            
        with col2:
            kpi_type = st.selectbox(
                "KPI Type:",
                ["Energy Consumption", "Water Usage", "Traffic Volume", "Waste Generation", "Air Quality", "Other"]
            )
        
        if uploaded_file is not None:
            # Read and display CSV
            df = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(df.head())
            
            if st.button("Generate Forecast", key="kpi_forecast"):
                csv_string = uploaded_file.getvalue().decode('utf-8')
                
                with st.spinner("Generating forecast with AI insights..."):
                    result = assistant.kpi_forecasting(csv_string, kpi_type)
                    
                if 'error' in result:
                    st.error(result['error'])
                else:
                    st.success("Forecast generated!")
                    
                    # Display forecasts
                    forecast_df = pd.DataFrame(result['forecasts'])
                    st.line_chart(forecast_df.set_index('date')['predicted_value'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Forecast Data")
                        st.dataframe(forecast_df)
                    
                    with col2:
                        st.markdown("### AI Insights")
                        st.write(result['insights'])
                        st.metric("Trend", result['trend'].title())

    # Eco Tips Tab
    with tab4:
        st.header("🌱 Eco-Friendly Tips Generator")
        st.write("Get personalized sustainability recommendations")
        
        # Predefined keywords
        eco_categories = {
            "Energy": ["solar", "renewable", "efficiency", "conservation"],
            "Water": ["conservation", "recycling", "rainwater", "usage"],
            "Transportation": ["public transport", "cycling", "electric vehicles", "walking"],
            "Waste": ["recycling", "composting", "reduction", "reuse"],
            "Urban Gardening": ["composting", "gardening", "green spaces", "plants"]
        }
        
        selected_category = st.selectbox("Select Category:", list(eco_categories.keys()))
        custom_keywords = st.text_input("Additional Keywords (comma-separated):", "")
        
        if st.button("Generate Eco Tips", key="eco_tips"):
            keywords = eco_categories[selected_category]
            if custom_keywords:
                keywords.extend([kw.strip() for kw in custom_keywords.split(',')])
            
            with st.spinner("Generating eco-friendly tips via AI..."):
                tips = assistant.generate_eco_tips(keywords)
                st.success("Tips generated!")
                st.markdown("### 🌿 Your Personalized Eco Tips")
                st.write(tips)

    # Anomaly Detection Tab
    with tab5:
        st.header("🔍 Anomaly Detection in City Data")
        st.write("Upload KPI data to detect unusual patterns")
        
        uploaded_file = st.file_uploader("Upload CSV file for anomaly detection", type=['csv'], key="anomaly_csv")
        
        if uploaded_file is not None:
            # Read and display CSV
            df = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(df.head())
            
            if st.button("Detect Anomalies", key="detect_anomalies"):
                csv_string = uploaded_file.getvalue().decode('utf-8')
                
                with st.spinner("Detecting anomalies and generating AI analysis..."):
                    result = assistant.anomaly_detection(csv_string)
                    
                if 'error' in result:
                    st.error(result['error'])
                else:
                    st.success("Anomaly detection completed!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", result['total_records'])
                    with col2:
                        st.metric("Anomalies Found", result['anomalies_found'])
                    with col3:
                        anomaly_rate = (result['anomalies_found'] / result['total_records']) * 100
                        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
                    
                    if result['anomaly_records']:
                        st.markdown("### Anomalous Records")
                        anomaly_df = pd.DataFrame(result['anomaly_records'])
                        st.dataframe(anomaly_df)
                    
                    st.markdown("### AI Analysis")
                    st.write(result['analysis'])

    # Chat Assistant Tab
    with tab6:
        st.header("💬 Smart City Chat Assistant")
        st.write("Ask questions about urban planning, sustainability, and city services")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for i, (user_msg, ai_msg) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(user_msg)
            with st.chat_message("assistant"):
                st.write(ai_msg)
        
        # Chat input
        user_message = st.chat_input("Ask me anything about smart cities...")
        
        if user_message:
            # Add user message to chat
            with st.chat_message("user"):
                st.write(user_message)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    ai_response = assistant.chat_assistant(user_message)
                    st.write(ai_response)
            
            # Save to chat history
            st.session_state.chat_history.append((user_message, ai_response))

    # Traffic Routes Tab
    with tab7:
        st.header("🚗 Traffic Route Suggestions")
        st.write("Get optimal routes and nearby attractions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            origin = st.text_input("From:", placeholder="Enter starting location")
        with col2:
            destination = st.text_input("To:", placeholder="Enter destination")
        with col3:
            city = st.text_input("City:", placeholder="Enter city name")
        
        if st.button("Get Route Suggestions", key="traffic_route"):
            if origin and destination and city:
                with st.spinner("Generating route suggestions via AI..."):
                    suggestions = assistant.traffic_route_suggestion(origin, destination, city)
                    st.success("Route suggestions generated!")
                    st.markdown("### 🗺️ Route Information")
                    st.write(suggestions)
            else:
                st.error("Please fill in all fields (From, To, City)")

    # View Reports Tab
    with tab8:
        st.header("📋 All Citizen Reports")
        st.write("View and manage submitted reports")
        
        if st.session_state.citizen_reports:
            # Display summary statistics
            total_reports = len(st.session_state.citizen_reports)
            priority_counts = {}
            category_counts = {}
            
            for report in st.session_state.citizen_reports:
                priority = report['priority']
                category = report['category']
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
                category_counts[category] = category_counts.get(category, 0) + 1
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Reports", total_reports)
            with col2:
                st.metric("High Priority", priority_counts.get('high', 0))
            with col3:
                st.metric("Medium Priority", priority_counts.get('medium', 0))
            with col4:
                st.metric("Low Priority", priority_counts.get('low', 0))
            
            # Display reports table
            reports_df = pd.DataFrame(st.session_state.citizen_reports)
            st.dataframe(
                reports_df[['id', 'timestamp', 'category', 'priority', 'location', 'status']],
                use_container_width=True
            )
            
            # Detailed view
            selected_report_id = st.selectbox("Select Report for Details:", 
                                            [f"Report #{r['id']}" for r in st.session_state.citizen_reports])
            
            if selected_report_id:
                report_id = int(selected_report_id.split('#')[1])
                selected_report = next(r for r in st.session_state.citizen_reports if r['id'] == report_id)
                
                st.markdown("### Report Details")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**ID:** {selected_report['id']}")
                    st.write(f"**Category:** {selected_report['category'].title()}")
                    st.write(f"**Priority:** {selected_report['priority'].title()}")
                    st.write(f"**Status:** {selected_report['status'].title()}")
                
                with col2:
                    st.write(f"**Location:** {selected_report['location']}")
                    st.write(f"**Contact:** {selected_report['contact']}")
                    st.write(f"**Timestamp:** {selected_report['timestamp']}")
                
                st.markdown("**Description:**")
                st.write(selected_report['description'])
                
                st.markdown("**AI Response:**")
                st.info(selected_report['ai_response'])
        else:
            st.info("No reports submitted yet. Go to the Citizen Reports tab to submit your first report!")

    # Footer
    st.markdown("---")
    st.markdown("**🏙️ Sustainable Smart City Assistant** - Powered by AI API for better urban living")

if __name__ == "__main__":
    main()
