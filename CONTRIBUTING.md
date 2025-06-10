# Contributing Guidelines

Welcome! 👋  
This document outlines our workflow and collaboration practices for this project. Please follow the steps below to ensure consistency and avoid merge conflicts.

---

## 🔁 Workflow Overview

We use a **feature-branch workflow**. All changes must be made in a branch and submitted via pull request (PR) to `main`.

---

## 🧩 Step-by-Step Workflow

### 1. Before You Start Work
Always pull the latest version of `master`:

```bash
git checkout master
git pull origin master
```

Then create a new branch for your work:

```bash
git checkout -b feature/your-branch-name
```

Use a clear name like `feature/data-cleaning`, `bugfix/typo-fix`, or `update/report-intro`.

---

### 2. Make Your Changes
Edit, add, or delete files as needed.

Stage and commit your changes with a meaningful message:

```bash
git add .
git commit -m "Add section to research report on PET preprocessing"
```

> 💡 Write short, clear commit messages in present tense.

---

### 3. Push and Create a Pull Request
Push your branch to GitHub:

```bash
git push origin feature/your-branch-name
```

Then go to GitHub, open a Pull Request (PR) from your branch into `master`, and add a short description of what you did.

---

### 4. After Review and Approval
Once reviewed, we’ll merge it into `master`.

To stay in sync, always run this before starting new work:

```bash
git checkout master
git pull origin master
```

---

## 🧠 Need Help?
If you're unsure about Git commands or run into issues, message **Jackson** on Discord — happy to help.

---

Thanks for keeping things clean and collaborative!
