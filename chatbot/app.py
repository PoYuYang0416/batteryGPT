import pandas as pd
import openai
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity

def count_tokens(text):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens
def split_content(input_string, tokens):
    """Splits a string into chunks based on a maximum token count. """

    MAX_TOKENS = tokens
    split_strings = []
    current_string = ""
    tokens_so_far = 0

    for word in input_string.split():
        # Check if adding the next word would exceed the max token limit
        if tokens_so_far + count_tokens(word) > MAX_TOKENS:
            # If we've reached the max tokens, look for the last dot or newline in the current string
            last_dot = current_string.rfind(".")
            last_newline = current_string.rfind("\n")

            # Find the index to cut the current string
            cut_index = max(last_dot, last_newline)

            # If there's no dot or newline, we'll just cut at the max tokens
            if cut_index == -1:
                cut_index = MAX_TOKENS

            # Add the substring to the result list and reset the current string and tokens_so_far
            split_strings.append(current_string[:cut_index + 1].strip())
            current_string = current_string[cut_index + 1:].strip()
            tokens_so_far = count_tokens(current_string)

        # Add the current word to the current string and update the token count
        current_string += " " + word
        tokens_so_far += count_tokens(word)

    # Add the remaining current string to the result list
    split_strings.append(current_string.strip())

    return split_strings

def add_similarity(df, given_embedding):
    def calculate_similarity(embedding):
        # Check if embedding is a string and convert it to a list of floats if necessary
        if isinstance(embedding, str):
            embedding = [float(x) for x in embedding.strip('[]').split(',')]
        return cosine_similarity([embedding], [given_embedding])[0][0]

    df['similarity'] = df['embedding'].apply(calculate_similarity)
    return df

def top_similar_entries(df, x=5):
    """
    Return the top x entries in the "org result" column based on the highest similarity values.

    :param df: The DataFrame containing the "similarity" and "Synthesis Information" columns.
    :param x: The number of top entries to return. Default is 3.
    :return: A string containing the top x entries in the "Synthesis Information" column, separated by new lines.
    """
    # Sort the DataFrame based on the "similarity" column in descending order
    sorted_df = df.sort_values(by="similarity", ascending=False)

    # Get the top x entries from the "Synthesis Information" column
    top_x_entries = sorted_df["summarized"].head(x).tolist()

    # Add separator line with MOF Name if x is equal or larger than 2
#     if x >= 2:
#         for i, entry in enumerate(top_x_entries):
#             mof_name = entry.split("\n")[0].replace("MOF Name: ", "")
#             separator = f"--- SECTION {i + 1}: {mof_name} ---"
#             top_x_entries[i] = separator + "\n" + entry

    # Join the entries together with new lines
    joined_entries = "\n".join(top_x_entries)

    return joined_entries

openai.api_key = "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"
def chatbot(question, past_user_messages=None, initial_context=None):    
    if past_user_messages is None:
        past_user_messages = []

    past_user_messages.append(question)

    file_name = "C:/Users/user/Desktop/battery NLP/chatbot/GPT4_JSON_embedded.csv" #synthesis information database with embedding
    df_with_emb = pd.read_csv(file_name)

    if initial_context is None:
        # Find the context based on the first question
        first_question = past_user_messages[0]
        question_return = openai.Embedding.create(model="text-embedding-ada-002", input=first_question)
        question_emb = question_return['data'][0]['embedding']

        df_with_emb_sim = add_similarity(df_with_emb, question_emb)
        num_paper = 5
        top_n_synthesis_str = top_similar_entries(df_with_emb_sim, num_paper)

        print("I have found below synthesis conditions and paper information based on your first question:")
        print("\n" + top_n_synthesis_str)
        initial_context = top_n_synthesis_str
    message_history = [
        {
            "role": "system",
            "content": "You are a battery assistant that specifically handles questions related to battery fabrication based on the papers you have reviewed. Answer the question based on the provided context." 
        },
        {
            "role": "user","content": "Battery paper context provided:"+ initial_context +"\nYour answer will be strictly based on the battery paper context provided" 
        
        },
    ]

    for user_question in past_user_messages:
        message_history.append({"role": "user", "content": user_question})

    response = openai.ChatCompletion.create(
        model='gpt-4',
        temperature=0,
        #max_tokens=2000,
        messages=message_history
    )

    answer = response.choices[0].message["content"]
    return answer, initial_context, past_user_messages



from flask import Flask, render_template, request, jsonify
past_user_messages = ""
initial_context= ""
app = Flask(__name__)
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/get_response", methods=["POST"])
def get_response():
    global past_user_messages
    global initial_context
    user_message = request.json["user_text"]
    if not past_user_messages :
        answer, initial_context, past_user_messages = chatbot(user_message)
    else:
        answer, _, past_user_messages = chatbot(user_message, past_user_messages, initial_context)
    return jsonify({"response": answer})
@app.route('/send-message', methods=['POST'])
def refresh():
    global past_user_messages
    global initial_context
    message_clear = request.get_json()
    if message_clear == "Clear":
        print("Clear")
        past_user_messages=""
        initial_context=""
    return jsonify({"past_user_messages": past_user_messages, "initial_context": initial_context})
if __name__ == "__main__":
    app.run(debug=True)

# first_question = "Please find a polymer layer on anode protection method and elaborate the fabrication process, strictly based on the context provided"
# follow_up_question = "Sure, let's dive into the details of the polymer layer anode protection method. Please provide a step-by-step explanation of the fabrication process involved