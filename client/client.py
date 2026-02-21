import requests
import urllib3

urllib3.disable_warnings()

SERVER_URL = "https://127.0.0.1:5000"

#def start_troubleshooting():


def troubleshot():
    """
    Send troubleshooting consent to server and handle server responses.

    Flow:
      1. Ask user for consent (yes/no).
      2. POST consent to /troubleshoot endpoint.
      3. Print any messages returned by server.
      4. If user says no, server is expected to return the next error message.
    """
    fb = input(
            "We need to read the current logs. Please accept the terms and conditions "
            "of reading: Type 'y', 'yes' or 'n', 'no': "
        ).strip().lower()

    if fb in ("y", "yes"):
        consent = "yes"
    
    else:
        return
    
    while True:
        try:
            print("starting troubleshooting.....")
            resp = requests.post(
                f"{SERVER_URL}/troubleshoot",
                json={"consent": consent},
                verify=False,
                timeout=300,
            )
        except Exception as e:
            print(f"Error communicating with server: {e}")
            return

        if resp.status_code != 200:
            print(f"Server returned status {resp.status_code}")
            print(resp.text)
            return

        try:
            data = resp.json()
        except ValueError:
            print("Invalid response from server (not JSON).")
            print(resp.text)
            return

        status = data.get("status")
        if status == "matched":
            message = data.get("message")
            print(message)
            #error = data.get("error")
            #print(error)
            requires_approval = data.get("requires_approval")
            print("requires_approval : ", requires_approval)
            if requires_approval == True:
                user_input = input(
                    "\nDo you want to apply this fix now? (yes/no): "
                ).strip().lower()
                if user_input in ("yes", "y"):
                    consent = "Fix_Error"
                else:
                    consent = "skip"

                # send user's decision back to server
                try:
                    print("Request send to fix the error : ", consent)
                    resp = requests.post(
                        f"{SERVER_URL}/troubleshoot",
                        json={"consent": consent},
                        verify=False,
                        timeout=300,
                    )
                    if resp.status_code != 200:
                        print(f"Server returned status {resp.status_code}")
                        print(resp.text)
                        return
                    if status == "Fixed":
                        print("Error is fixed. Please check again")
                        return
                except Exception as e:
                    print(f"Error communicating with server: {e}")
                    return

                if resp.status_code != 200:
                    print(f"Server returned status {resp.status_code}")
                    print(resp.text)
                    return

                try:
                    data = resp.json()
                except ValueError:
                    print("Invalid response from server (not JSON).")
                    print(resp.text)
                    return
            break
        if status == "done":
            print("No more error present.")
            break
        if status == "fixed":
            print("Error is fixed. Please check again")
            break

        # Expecting server to return at least a message, and optionally a flag to continue
        message = data.get("message")
        if message:
            print(message)
        
        log = data.get("log")
        if log:
            print(log)
        
        # Ask user whether to fetch next error or end troubleshooting
        user_choice = input(
            "\nType yes if you want to troubleshoot this error? :\nType 'next' to read the next error, or 'end' to stop troubleshooting: "
        ).strip().lower()

        while True:
            if user_choice in ("end", "e", "q", "quit"):
                consent = "no"
                break
            elif user_choice in ("next", "n"):
                consent = "yes"
                break
            elif user_choice in ("yes", "y"):
                consent = "troubleshoot"
                break
            else:
                continue


        # If server says no more troubleshooting is needed, break
        if not data.get("continue", False):
            break

        # If server wants to ask again (e.g., next error message after 'no'), loop continues


def chat():
    print("🔐 Secure Chat Client")
    print("Type 'q' or 'quit' to exit\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in ("q", "quit"):
            print("👋 Bye!")
            break

        if query.lower() in ("troubleshoot"):
            troubleshot()
            continue

        try:
            resp = requests.post(
                f"{SERVER_URL}/api",
                json={"text": query},
                verify=False,
                timeout=300,
            )
        except Exception as e:
            print(f"Error communicating with server: {e}")
            print("-" * 50)
            continue

        if resp.status_code != 200:
            print(f"Server returned status {resp.status_code}")
            print(resp.text)
            print("-" * 50)
            continue

        try:
            res = resp.json()
        except ValueError:
            print("Invalid response from server (not JSON).")
            print(resp.text)
            print("-" * 50)
            continue

        # ✅ ONLY ask feedback if the bot answered
        status = res.get("status", "unknown")

        print("\nBot:", res.get("answer", "No response"))
        
        # Ask feedback ONLY if model answered confidently
        if status in ("answered", "pdf_answer"):
            fb = input("Is this answer correct? (yes/no): ").strip().lower()
        
            requests.post(
                f"{SERVER_URL}/feedback",
                json={
                    "question": query,
                    "answer": res["answer"],
                    "feedback": fb
                },
                verify=False,
                timeout=300,
            )
        
        elif status == "unanswered":
            print("📝 Question saved for future training.")
        
        elif status == "invalid":
            print("⚠️ Please ask a longer or clearer question.")
        
        elif status == "error":
            print("❌ Server error:", res.get("details"))

     
        print("-" * 50)
        


if __name__ == "__main__":
    chat()