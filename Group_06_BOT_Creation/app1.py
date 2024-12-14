import streamlit as st
import openai
from pinecone import Pinecone

# Initialize OpenAI API
openai.api_key = "Your API key"

# Initialize Pinecone
api_key = "pcsk_EB1ry_N9StDcXxKY6MEaxKkBRuZzczM3pYn9MUNC15jUC4p2XALtwpo7eocAcsGYpV3c5"
host = "https://career-guidance-index-rxt2r7i.svc.aped-4627-b74a.pinecone.io"  # Replace with your Pinecone host
pc = Pinecone(api_key=api_key)
index_name = "career-guidance"

# Check if index exists and create if necessary
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Adjust dimension to your embedding size
        metric="cosine"
    )

# Connect to the index
index = pc.Index(name=index_name, host=host)

# Streamlit App
st.title("Career Guidance Bot")
st.write("Welcome to the Career Guidance Application powered by Streamlit!")

# Initialize session state to maintain chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful career guidance assistant. How can I help you today?"}
    ]

# Display chat history
st.write("### Chat History")
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"**User:** {message['content']}")
    elif message["role"] == "assistant":
        st.write(f"**Bot:** {message['content']}")

# Input for user query
query = st.text_input("Enter your query:")

if st.button("Send"):
    if query:
        # Add user query to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        # Generate embedding for the query
        query_embedding = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002"
        )['data'][0]['embedding']

        # Query Pinecone for relevant context
        query_result = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )

        # Combine contexts
        context = "\n\n".join([match['metadata']['text'] for match in query_result['matches']])

        # Generate GPT-4 response
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=st.session_state.messages + [
                {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
            ],
            max_tokens=300,
            temperature=0.5
        )

        # Extract the response message
        bot_response = response['choices'][0]['message']['content'].strip()

        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

        # Display the response
        st.write(f"**Bot:** {bot_response}")
    else:
        st.write("Please enter a query!")
