"""
Prompt Manager for Sokoban Agent
Handles loading and managing different versions of system prompts from YAML files
"""

import yaml
import os
from typing import Dict, List, Optional
from pathlib import Path


class PromptManager:
    """Manages different versions of system prompts for the Sokoban agent"""
    
    def __init__(self, prompts_file: str = "prompts/sokoban_prompts.yaml"):
        """
        Initialize the prompt manager
        
        Args:
            prompts_file: Path to the YAML file containing prompts
        """
        self.prompts_file = prompts_file
        self.prompts = {}
        self.config = {}
        self.load_prompts()
    
    def load_prompts(self) -> None:
        """Load prompts from the YAML file"""
        try:
            prompts_path = Path(self.prompts_file)
            if not prompts_path.exists():
                raise FileNotFoundError(f"Prompts file not found: {self.prompts_file}")
            
            with open(prompts_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
                
            self.prompts = data.get('prompts', {})
            self.config = data.get('config', {})
            
            print(f"✅ Loaded {len(self.prompts)} prompt versions")
            for version, prompt_data in self.prompts.items():
                print(f"   - {version}: {prompt_data.get('name', 'Unnamed')}")
                
        except Exception as e:
            print(f"❌ Error loading prompts: {e}")
            # Fallback to default prompt
            self.prompts = {
                'default': {
                    'name': 'Default Sokoban Solver',
                    'description': 'Fallback prompt',
                    'content': 'You are solving a Sokoban puzzle. Push boxes to targets.'
                }
            }
            self.config = {'default_version': 'default'}
    
    def get_prompt(self, version: Optional[str] = None) -> str:
        """
        Get a specific prompt version
        
        Args:
            version: Version to get (e.g., 'v1', 'v2'). If None, uses default.
            
        Returns:
            The prompt content as a string
        """
        if version is None:
            version = self.config.get('default_version', 'default')
        
        if version is None or version not in self.prompts:
            print(f"⚠️  Prompt version '{version}' not found, using default")
            version = self.config.get('default_version', 'default')
            if version is None:
                version = 'default'
        
        prompt_data = self.prompts.get(version, {})
        return prompt_data.get('content', 'Default prompt content')
    
    def get_prompt_info(self, version: Optional[str] = None) -> Dict:
        """
        Get information about a prompt version
        
        Args:
            version: Version to get info for. If None, uses default.
            
        Returns:
            Dictionary with prompt information
        """
        if version is None:
            version = self.config.get('default_version', 'default')
        
        if version is None or version not in self.prompts:
            return {'name': 'Unknown', 'description': 'Version not found'}
        
        return self.prompts[version]
    
    def list_available_versions(self) -> List[str]:
        """Get list of available prompt versions"""
        return list(self.prompts.keys())
    
    def get_config(self) -> Dict:
        """Get the configuration settings"""
        return self.config.copy()
    
    def switch_prompt_version(self, current_version: str, failure_count: int = 0) -> str:
        """
        Automatically switch to a different prompt version based on failure count
        
        Args:
            current_version: Current prompt version
            failure_count: Number of consecutive failures
            
        Returns:
            New prompt version to try
        """
        if not self.config.get('auto_switch_on_failure', False):
            return current_version
        
        max_retries = self.config.get('max_retries_per_version', 3)
        if isinstance(max_retries, int) and failure_count < max_retries:
            return current_version
        
        # Get available versions excluding current
        available_versions = self.list_available_versions()
        if current_version in available_versions:
            available_versions.remove(current_version)
        
        if not available_versions:
            return current_version
        
        # Simple round-robin selection
        next_version = available_versions[0]
        print(f"🔄 Switching from '{current_version}' to '{next_version}' after {failure_count} failures")
        return next_version
    
    def add_prompt_version(self, version: str, name: str, description: str, content: str) -> None:
        """
        Add a new prompt version
        
        Args:
            version: Version identifier (e.g., 'v5')
            name: Display name for the prompt
            description: Description of the prompt
            content: The actual prompt content
        """
        self.prompts[version] = {
            'name': name,
            'description': description,
            'content': content
        }
        print(f"✅ Added new prompt version: {version} - {name}")
    
    def save_prompts(self) -> None:
        """Save current prompts back to the YAML file"""
        try:
            data = {
                'prompts': self.prompts,
                'config': self.config
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.prompts_file), exist_ok=True)
            
            with open(self.prompts_file, 'w', encoding='utf-8') as file:
                yaml.dump(data, file, default_flow_style=False, allow_unicode=True)
            
            print(f"✅ Saved prompts to {self.prompts_file}")
            
        except Exception as e:
            print(f"❌ Error saving prompts: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Test the prompt manager
    manager = PromptManager()
    
    print("\n📋 Available prompt versions:")
    for version in manager.list_available_versions():
        info = manager.get_prompt_info(version)
        print(f"   {version}: {info['name']} - {info['description']}")
    
    print(f"\n🔧 Configuration:")
    config = manager.get_config()
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print(f"\n📝 Default prompt preview:")
    default_prompt = manager.get_prompt()
    print(default_prompt[:200] + "..." if len(default_prompt) > 200 else default_prompt) 