from whitelist_manager import WhitelistManager
import os

def main():
    if not os.path.exists("legit_pages"):
        print("legit_pages directory not found!")
        return
        
    print("Initializing Whitelist Manager...")
    wm = WhitelistManager()
    
    print("Building whitelist from legit_pages...")
    wm.build_from_directory("legit_pages")
    
    print("Done!")

if __name__ == "__main__":
    main()
