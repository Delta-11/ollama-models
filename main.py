import requests
import json

class DeepSeekChat:
    def __init__(self, model="deepseek-r1:7b"):
        self.model = model
        self.conversation_history = []

    def query(self, prompt):
        # Add the user's input to the conversation history
        self.conversation_history.append({"role": "user", "content": prompt})

        # Construct the full prompt
        context = "\n".join(f"{entry['role']}: {entry['content']}" for entry in self.conversation_history)

        url = "http://localhost:11434/api/generate"
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "prompt": context
        }

        try:
            # Make a streaming request
            response = requests.post(url, headers=headers, json=data, stream=True)
            if response.status_code == 200:
                print("Response:", end=" ")  # Initialize output on the same line
                response_text = ""
                for line in response.iter_lines():
                    if line:
                        chunk = line.decode("utf-8")
                        try:
                            chunk_data = json.loads(chunk)
                            chunk_response = chunk_data.get("response", "")
                            print(chunk_response, end="", flush=True)
                            response_text += chunk_response

                            if chunk_data.get("done", False):
                                break
                        except ValueError as e:
                            print(f"\nUnable to parse chunk: {chunk} - Error: {e}")
                
                # Add the model's response to the conversation history
                self.conversation_history.append({"role": "model", "content": response_text})
                print()  # Print a newline after the response
            else:
                print(f"API request failed with status code {response.status_code}")
        except Exception as e:
            print(f"Error processing request:\n{e}")

# Example usage
chat = DeepSeekChat()
chat.query("Explain the concept of reinforcement learning.")
