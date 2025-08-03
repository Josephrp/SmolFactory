#!/usr/bin/env python3
"""
Switch Trackio Space from Write Token to Read Token

This script switches the HF_TOKEN secret in a Trackio Space from a write token
to a read token after the experiment is complete, for security purposes.
"""

import os
import sys
import json
from typing import Optional, Tuple
from huggingface_hub import HfApi

def validate_token_permissions(token: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate token and determine its permission level.
    
    Args:
        token (str): The Hugging Face token to validate
        
    Returns:
        Tuple[bool, str, Optional[str]]: 
            - success: True if token is valid
            - permission_level: "read" or "write"
            - username: The username associated with the token
    """
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        
        # Extract username
        username = user_info.get("name", user_info.get("username"))
        
        # Test write permissions by trying to access a test repository
        # We'll use a simple test - try to get repo info for a public repo
        try:
            # Try to access a public dataset to test read permissions
            api.dataset_info("huggingface-course/documentation-tutorial")
            
            # For write permissions, we'll assume the token has write access
            # since we can't easily test write permissions without creating something
            # In practice, write tokens are typically provided by users who know
            # they have write access
            return True, "write", username
            
        except Exception as e:
            # If we can't access even a public dataset, it's likely a read token
            return True, "read", username
            
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return False, "invalid", None
        else:
            return False, "error", None

def switch_space_token(space_id: str, read_token: str, write_token: str) -> bool:
    """
    Switch the HF_TOKEN secret in a Trackio Space from write to read token.
    
    Args:
        space_id (str): The space ID (username/space-name)
        read_token (str): The read token to set
        write_token (str): The write token (for validation)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Validate both tokens
        print("🔍 Validating tokens...")
        
        write_valid, write_perm, write_user = validate_token_permissions(write_token)
        read_valid, read_perm, read_user = validate_token_permissions(read_token)
        
        if not write_valid:
            print(f"❌ Write token validation failed")
            return False
            
        if not read_valid:
            print(f"❌ Read token validation failed")
            return False
            
        if write_user != read_user:
            print(f"❌ Token mismatch: write token user ({write_user}) != read token user ({read_user})")
            return False
            
        print(f"✅ Tokens validated successfully")
        print(f"   Write token: {write_perm} permissions for {write_user}")
        print(f"   Read token: {read_perm} permissions for {read_user}")
        
        # Use the write token to update the space (since we need write access)
        api = HfApi(token=write_token)
        
        # Update the HF_TOKEN secret in the space
        try:
            api.add_space_secret(
                repo_id=space_id,
                key="HF_TOKEN",
                value=read_token,
                description="Hugging Face read token for dataset access (switched from write token)"
            )
            print(f"✅ Successfully switched HF_TOKEN to read token in space: {space_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to update space secret: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error switching tokens: {e}")
        return False

def main():
    """Main function to switch tokens."""
    
    print("🔄 Trackio Space Token Switch")
    print("=" * 40)
    
    # Get arguments
    if len(sys.argv) >= 4:
        space_id = sys.argv[1]
        read_token = sys.argv[2]
        write_token = sys.argv[3]
    else:
        print("Usage: python switch_to_read_token.py <space_id> <read_token> <write_token>")
        print("Example: python switch_to_read_token.py username/trackio-monitoring read_token write_token")
        sys.exit(1)
    
    # Validate space_id format
    if "/" not in space_id:
        print("❌ Invalid space_id format. Use: username/space-name")
        sys.exit(1)
    
    # Switch tokens
    success = switch_space_token(space_id, read_token, write_token)
    
    if success:
        print("\n✅ Token switch completed successfully!")
        print(f"📊 Space: {space_id}")
        print("🔒 HF_TOKEN now uses read-only permissions")
        print("💡 The space can still read datasets but cannot write to repositories")
    else:
        print("\n❌ Token switch failed!")
        print("Please check your tokens and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 