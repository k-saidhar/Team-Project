# ⚡ SIMPLE UBUNTU SETUP - DO THIS NOW

## 🎯 What You Need to Do

Ubuntu is installed but needs manual setup. Follow these exact steps:

### Step 1: Open Ubuntu (Choose ONE method)

**Method A: From Start Menu**
1. Press `Windows Key`
2. Type: **Ubuntu**
3. Click **Ubuntu** app

**Method B: From PowerShell**
1. Open new PowerShell window
2. Type: `ubuntu`
3. Press Enter

### Step 2: Create Account

When Ubuntu opens, you'll see:
```
Installing, this may take a few minutes...
Please create a default UNIX user account...
Enter new UNIX username:
```

**Enter these exactly:**
- Username: `Phishsim`
- Password: `Cyber@123` (you won't see it as you type)
- Confirm: `Cyber@123`

### Step 3: Run Setup Commands

After account creation, you'll see a prompt like: `Phishsim@DESKTOP:~$`

**Copy and paste these commands ONE BY ONE:**

```bash
# Update packages
sudo apt update

# Install dependencies (enter password: Cyber@123 when asked)
sudo apt install -y g++ liblzma-dev libxxhash-dev build-essential

# Go to project
cd /mnt/c/Users/Nagireddy123/Desktop/Project-f/phishsim

# Build C++ worker
bash build.sh

# Run clustering
python fast_cluster.py
```

---

## ✅ Expected Output

After `bash build.sh`:
```
Building ultra-fast NCD worker...
Build successful! → ./ncd_worker
```

---

## ❌ If You Get Errors

### "sudo: command not found"
- You're not in Ubuntu. Type `ubuntu` or `wsl` to enter

### "bash: build.sh: No such file"
- Run: `cd /mnt/c/Users/Nagireddy123/Desktop/Project-f/phishsim`
- Then: `ls build.sh` (should show the file)

### "Permission denied"
```bash
chmod +x build.sh
bash build.sh
```

---

## 🚀 That's It!

Once the build completes, you're done! The C++ worker is ready and you can run `python fast_cluster.py` anytime.
