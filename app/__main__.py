# app/__main__.py

from settings import Settings  # 👈 This will load required env variables
from agent import root_agent   # 👈 Make sure your `root_agent` is imported here

def main():
    # Load deployment settings (project, region, vertex flag, etc.)
    settings = Settings()
    
    # Optionally: Print confirmation (debug purposes)
    print("Settings loaded:", settings)

    # No other startup code is needed — the Agent Engine will use root_agent
    # Just make sure root_agent is defined at module level (in agent.py)

if __name__ == "__main__":
    main()
