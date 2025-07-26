#!/usr/bin/env python3
"""
Hugging Face Token Validation Script

This script validates a Hugging Face token and retrieves the associated username
using the huggingface_hub Python API.
"""

import sys
import os
from typing import Optional, Tuple
from huggingface_hub import HfApi, login
import json

def validate_hf_token(token: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate a Hugging Face token and return the username.
    
    Args:
        token (str): The Hugging Face token to validate
        
    Returns:
        Tuple[bool, Optional[str], Optional[str]]: 
            - success: True if token is valid, False otherwise
            - username: The username associated with the token (if valid)
            - error_message: Error message if validation failed
    """
    try:
        # Set the token as environment variable
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        
        # Create API client
        api = HfApi()
        
        # Try to get user info - this will fail if token is invalid
        user_info = api.whoami()
        
        # Extract username from user info
        username = user_info.get("name", user_info.get("username"))
        
        if not username:
            return False, None, "Could not retrieve username from token"
            
        return True, username, None
        
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return False, None, "Invalid token - unauthorized access"
        elif "403" in error_msg:
            return False, None, "Token lacks required permissions"
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            return False, None, f"Network error: {error_msg}"
        else:
            return False, None, f"Validation error: {error_msg}"

def main():
    """Main function to validate token from command line or environment."""
    
    # Get token from command line argument or environment variable
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    if not token:
        print(json.dumps({
            "success": False,
            "username": None,
            "error": "No token provided. Use as argument or set HF_TOKEN environment variable."
        }))
        sys.exit(1)
    
    # Validate token
    success, username, error = validate_hf_token(token)
    
    # Return result as JSON for easy parsing
    result = {
        "success": success,
        "username": username,
        "error": error
    }
    
    print(json.dumps(result))
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 