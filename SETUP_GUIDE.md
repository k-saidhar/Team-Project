# 🚀 WSL Installation - Step by Step Guide

## ⚠️ IMPORTANT: You Need Administrator Privileges

The PowerShell script requires administrator access. Here's how to proceed:

---

## 📋 Method 1: Manual Installation (Recommended - Easiest)

### Step 1: Open PowerShell as Administrator

1. Press `Windows Key`
2. Type: **PowerShell**
3. **Right-click** on "Windows PowerShell"
4. Click **"Run as Administrator"**
5. Click **Yes** when prompted

### Step 2: Navigate to Project Directory

```powershell
cd C:\Users\Nagireddy123\Desktop\Project-f\phishsim
```

### Step 3: Install WSL

```powershell
wsl --install
```

**What happens:**
- Downloads and installs WSL
- Installs Ubuntu (default distribution)
- Takes 5-10 minutes depending on internet speed

### Step 4: Restart Your Computer

```powershell
# The system will prompt you to restart
# Or manually restart your computer
```

### Step 5: First-Time Ubuntu Setup (After Restart)

1. **Ubuntu will open automatically** after restart
2. Wait 1-2 minutes for installation to complete
3. **Create a username** (e.g., `yourname`)
4. **Create a password** (you won't see it as you type - this is normal)
5. **Confirm password**

### Step 6: Install Build Dependencies

In the Ubuntu terminal that just opened, run these commands:

```bash
# Update package list
sudo apt update

# Install dependencies (enter your password when prompted)
sudo apt install -y g++ liblzma-dev libxxhash-dev build-essential
```

### Step 7: Navigate to Project and Build

```bash
# Navigate to your project
cd /mnt/c/Users/Nagireddy123/Desktop/Project-f/phishsim

# Build the C++ worker
bash build.sh
```

**Expected output:**
```
Building ultra-fast NCD worker...
Build successful! → ./ncd_worker
```

### Step 8: Run Fast Cluster

```bash
python fast_cluster.py
```

---

## 📋 Method 2: Using the PowerShell Script

If you prefer the automated script:

### Step 1: Open PowerShell as Administrator
(Same as Method 1, Step 1)

### Step 2: Navigate and Run Script

```powershell
cd C:\Users\Nagireddy123\Desktop\Project-f\phishsim
.\install_wsl.ps1
```

### Step 3: Restart When Prompted

### Step 4: After Restart, Run Setup Script

Open **normal PowerShell** (not admin) and run:

```powershell
cd C:\Users\Nagireddy123\Desktop\Project-f\phishsim
.\setup_wsl_and_build.ps1
```

---

## 🔧 Troubleshooting

### "Scripts are disabled on this system"

**Already fixed!** I ran this command for you:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
```

If you still see this error in a new PowerShell window, run the command again.

### "This script must be run as Administrator"

You need to:
1. Close current PowerShell
2. Right-click PowerShell → **Run as Administrator**
3. Try again

### "wsl: command not found" (after restart)

- Make sure you restarted your computer
- Check Windows version: WSL requires Windows 10 version 2004+ or Windows 11
- Check version: `winver` in Run dialog (Windows + R)

### Build fails with "bash: build.sh: No such file or directory"

Make sure you're in the correct directory:
```bash
cd /mnt/c/Users/Nagireddy123/Desktop/Project-f/phishsim
pwd  # Should show the path above
ls build.sh  # Should show the file
```

---

## ✅ Verification Checklist

After setup, verify everything works:

```bash
# In Ubuntu/WSL terminal

# 1. Check if you're in the right directory
pwd
# Should output: /mnt/c/Users/Nagireddy123/Desktop/Project-f/phishsim

# 2. Check if C++ worker was built
ls -lh ncd_worker
# Should show the executable file

# 3. Test the worker
./ncd_worker
# Should show usage/error (this is expected)

# 4. Check Python
python --version
# Should show Python version

# 5. Check if HTML files exist
ls rendered_pages_parallel/*.html | head -5
# Should list some HTML files
```

---

## 🎯 Quick Reference

### Open WSL/Ubuntu Terminal
- **From Start Menu:** Type "Ubuntu" → Click "Ubuntu"
- **From PowerShell:** Type `wsl` and press Enter
- **From File Explorer:** In project folder, type `wsl` in address bar

### Navigate to Project in WSL
```bash
cd /mnt/c/Users/Nagireddy123/Desktop/Project-f/phishsim
```

### Run Clustering
```bash
python fast_cluster.py
```

### Exit WSL
```bash
exit
```

---

## 📞 Need Help?

If you encounter any issues:

1. Check the error message carefully
2. Look in the Troubleshooting section above
3. Check `SETUP_GUIDE.md` for more detailed information
4. Verify your Windows version supports WSL (Windows 10 2004+ or Windows 11)

---

## 🔄 What to Do Right Now

**Choose one method and follow the steps:**

✅ **Method 1 (Manual)** - More control, step-by-step
✅ **Method 2 (Script)** - Automated, faster

**Both methods require:**
1. Administrator PowerShell
2. Computer restart
3. ~15-20 minutes total time
