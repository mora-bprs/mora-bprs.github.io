---
title: "How to get started"
date: 2024-03-03
authors:
  - name: imfing
    link: https://go.io

tags:
  - Markdown
  - Example
  - Guide 

draft: false
---

## How to get started

1. **Clone the Repository:**
   - Developers should start by cloning the repository to their local machine:
     ```
     git clone https://github.com/mora-bprs/mora-bprs.github.io.git
     ```

2. **Navigate to the Hugo Site Directory:**
   - Once cloned, developers should navigate to the directory where the Hugo site is located:
     ```
     cd mora-bprs.github.io
     ```

3. **Install Hugo:**
   - Developers need to have Hugo installed on their local machine. They can follow the installation instructions provided on the [official Hugo website](https://gohugo.io/installation/).

4. **Start Editing the Site:**
   - Developers can start editing the Hugo site by modifying the content files located in the `content` directory. They can use Markdown syntax to write content and make changes as needed.

   ``````

5. **Preview Changes Locally:**
   - To preview their changes locally, developers can run the Hugo development server:
     ```
     hugo server -D
     ```
     This command starts a local web server, and developers can view the site in their web browser at `http://localhost:1313`.

6. **Commit Changes to the "dev" Branch:**
   - After making changes, developers should commit their changes to the "dev" branch:
     ```
     git checkout dev
     git add .
     git commit -m "Description of changes made"
     git push origin dev
     ```

7. **Submit Pull Requests (Optional):**
   - If your workflow involves code reviews, developers can submit pull requests from the "dev" branch to other branches (e.g., "main" for production). This allows for review and collaboration before merging changes.

8. **Deploy to Production (Optional):**
   - Once changes are ready to be deployed to production, you can merge the "dev" branch into the appropriate branch (e.g., "main") to trigger a deployment.

By documenting these steps, developers will have a clear understanding of how to clone, edit, and push updates to the Hugo site hosted in the "mora-bprs.github.io" repository, helping to streamline the development process.