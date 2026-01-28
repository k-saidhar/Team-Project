# Ubuntu Account Setup - Next Steps

## ✅ Ubuntu is Installing!

Ubuntu installation is in progress and waiting for your input.

---

## 🎯 What You Need to Do RIGHT NOW

### Step 1: Look at Your PowerShell Window

Ubuntu should be prompting you for account creation:

1. **Enter a username** (e.g., `nagireddy` or any name you prefer)
   - Use lowercase letters
   - No spaces
   - Press Enter

2. **Enter a password**
   - **IMPORTANT:** You won't see the password as you type (this is normal for Linux)
   - Type carefully and press Enter

3. **Confirm password**
   - Type the same password again
   - Press Enter

---

## 📋 After Account Creation

Once you see the Ubuntu prompt (looks like `nagireddy@DESKTOP:~$`), run these commands **one by one**:

### Command 1: Update Package List
```bash
sudo apt update
```
- Enter your password when prompted
- Wait for it to complete (~30 seconds)

### Command 2: Install Build Dependencies
```bash
sudo apt install -y g++ liblzma-dev libxxhash-dev build-essential
```
- This will take 2-3 minutes
- Wait for it to complete

### Command 3: Navigate to Project
```bash
cd /mnt/c/Users/Nagireddy123/Desktop/Project-f/phishsim
```

### Command 4: Build the C++ Worker
```bash
bash build.sh
```

**Expected output:**
```
Building ultra-fast NCD worker...
Build successful! → ./ncd_worker
```

### Command 5: Run Fast Cluster
```bash
python fast_cluster.py
```

---

## 🔍 Troubleshooting

### "sudo: command not found"
- You're not in Ubuntu. Type `wsl` to enter Ubuntu.

### "bash: build.sh: No such file or directory"
- Make sure you ran Command 3 (cd to project directory)
- Check with: `pwd` (should show the project path)

### "Permission denied"
```bash
chmod +x build.sh
bash build.sh
```

---

## ✅ Verification

After building, verify everything works:

```bash
# Check if worker was built
ls -lh ncd_worker

# Test it
./ncd_worker
```

---

## 📞 Let Me Know

After you complete the account setup and run the commands, let me know:
- ✅ If the build was successful
- ❌ If you encountered any errors (copy the error message)
