import os
import sys
import json
import argparse
import subprocess

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    messages = [{"role": "user", "content": args.p}]

    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "description": "Read and return the contents of a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "The path to the file to read",
                                }
                            },
                            "required": ["file_path"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "Write",
                        "description": "Write content to a file",
                        "parameters": {
                            "type": "object",
                            "required": ["file_path", "content"],
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "The path of the file to write to",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "The content to write to the file",
                                },
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "description": "Execute a shell command",
                        "parameters": {
                            "type": "object",
                            "required": ["command"],
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "The command to execute",
                                }
                            },
                        },
                    },
                },
            ],
        )

        messages.append(chat.choices[0].message)

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        tool_calls = chat.choices[0].message.tool_calls

        if tool_calls is None or len(tool_calls) == 0:
            messages.append(chat.choices[0].message.content)
            print(chat.choices[0].message.content)
            break
        else:
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                if tool_name == "Read":
                    tool_args = json.loads(tool_call.function.arguments)

                    with open(tool_args.get("file_path"), "r") as f:
                        content = f.read()

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": content,
                        }
                    )
                elif tool_name == "Write":
                    tool_args = json.loads(tool_call.function.arguments)

                    with open(tool_args.get("file_path"), "w") as f:
                        content = f.write(tool_args.get("content"))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_args.get("content"),
                        }
                    )
                elif tool_name == "Bash":
                    tool_args = json.loads(tool_call.function.arguments)
                    content = subprocess.run(
                        tool_args.get("command"),
                        shell=True,
                        capture_output=True,
                        text=True,
                    )
                    if content.stdout:
                        print(content.stdout)
                    else:
                        print(content.stderr)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": content.stdout
                            if content.stdout
                            else content.stderr,
                        }
                    )

    print("Logs from your program will appear here!", file=sys.stderr)


if __name__ == "__main__":
    main()
