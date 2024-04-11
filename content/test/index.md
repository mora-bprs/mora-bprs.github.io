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
   - you should start by cloning the repository to their local machine:

     ```shell
     git clone https://github.com/mora-bprs/mora-bprs.github.io.git
     ```

2. **Navigate to the Hugo Site Directory:**
   - Once cloned, you should navigate to the directory where the Hugo site is located:

     ```shell
     cd mora-bprs.github.io
     ```

3. **Install Hugo:**
   - You need to have Hugo installed on their local machine. They can follow the installation instructions provided on the [official Hugo website](https://gohugo.io/installation/).

4. **Start Editing the Site:**
   - Refer to this [hugo theme documentation](https://imfing.github.io/hextra/docs/) to edit the pages. 
   - You can start editing the Hugo site by modifying the content files located in the `content` directory. They can use Markdown syntax to write content and make changes as needed.

   first create a folder with the slug name in the content folder, then create an index.md file in that folder and write the content in it.

   template for index.md file:

   ```markdown
   ---
   title: "Title of the page"
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

   ## Sample Markdown content

   Here write the content of the page in markdown syntax.
   ```

5. **Preview Changes Locally:**
   - To preview their changes locally, you can run the Hugo development server:

   ```shell
   hugo server -D
   ```

     This command starts a local web server, and you can view the site in their web browser at `http://localhost:1313`.

6. **Generate the changes locally:**
   - To generate the changes locally, you can run the Hugo command:

   ```shell
   hugo --destination docs
   ```

     This command generates the static files in the `docs` directory.

7. **Commit Changes to the "dev" Branch:**

   - Create a branch named `dev` when you run this for the first time.

     ```shell
     git checkout -b dev
     ```

   - If this is not your first time working with `dev` branch do the following. 
   - After making changes, you should commit their changes to the "dev" branch:

     ```shell
     git checkout dev
     git add .
     git commit -m "Description of changes made"
     git push origin dev
     ```
