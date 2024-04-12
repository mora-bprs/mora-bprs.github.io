# Documentation Site for BPRS Projects

This site holds the documentation for the research and literary reviews done by Mora BPRS.

## Get Started Now
1. **Instll Hugo** - [official Hugo website](https://gohugo.io/installation/).
2. **Clone with submodules and navigate**
   ```shell
   git clone --recurse-submodules https://github.com/mora-bprs/mora-bprs.github.io.git && cd mora-bprs.github.io
   #or
   gh repo clone mora-bprs/mora-bprs.github.io -- --recurse-submodules && cd mora-bprs.github.io
   ```
3. **Create a branch**
     ```shell
     git checkout -b <newbranchname>
     # example: git checkout -b thuva
     ```
4. **Make changes and Push**
    - edit files inside `docs/` directory referring [hugo theme documentation](https://imfing.github.io/hextra/docs/) 
    - check using the following command, locally hosted in `localhost:1313`
    ```shell
    hugo server -D
    ```
    - upload your changes to repo when you are confident and make a pull request
    ```shell
    git add .
    git commit -m "Description of changes made"
    git push origin <remotebranchname>
    ```

## Detailed Guide

- This guide assumes that you have previous experience with handling banches and private repos, if you want a more beginner friendly approach use the github commandline utility `gh` which will setup all the authetications and remote url handling for you.
- We are using the `hextra` template for our documentation site. And choosing the git submodule route to install the theme.
- Follow the steps to get the site working first and then add the content.

1. **Install Hugo**

   - Hugo is a fast Static Site generator written in golang, hence you have to install it to compile your templates to web format.
   - You can follow the installation instructions provided on the [official Hugo website](https://gohugo.io/installation/).

2. **Clone the Repository with submodules and Navigate to the Directory**

   ```shell
   git clone --recurse-submodules https://github.com/mora-bprs/mora-bprs.github.io.git && cd mora-bprs.github.io
   ```

3. **Start Editing the Site:**

   - Refer to this [hugo theme documentation](https://imfing.github.io/hextra/docs/) to edit the pages.
   - You can start editing the Hugo site by modifying the content files located in the `content` directory. They can use Markdown syntax to write content and make changes as needed.
   - First create a folder with the slug name in the content folder, then create an index.md file in that folder and write the content in it.
   - `hugo server -D` will give a preview of your site in `localhost:1313`

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

### Useful Commands

1. **Preview Changes Locally:**

   - To preview their changes locally, you can run the Hugo development server:

   ```shell
   hugo server -D
   #or
   hugo server --buildDrafts --disableFastRender
   ```

   This command starts a local web server, and you can view the site in their web browser at `http://localhost:1313`.

2. **Generate the changes locally:**

   - To generate the changes locally, you can run the Hugo command:

   ```shell
   hugo --destination docs
   ```

   This command generates the static files in the `docs` directory.

3. **Commit Changes to the respective Branch:**

   - Create a new branch when you run this for the first time.

     ```shell
     git checkout -b <newbranchname>
     # example: git checkout -b thuva
     ```

   - If this is not your first time working with `dev` branch do the following.
   - After making changes, you should commit their changes to the "dev" branch:
   - When multiple people are working it is recommended to create a branch using your name or something unique to your team and commit to that branch to avoid merge conflicts and remote HEAD conflicts.

     - Sasika: dev branch
     - Thuva: thuva branch

     ```shell
     git checkout <branchname>
     git add .
     git commit -m "Description of changes made"
     git push origin <remotebranchname>
     ```

4. **Updating Submodules**

```shell
git submodule update --remote

```

### Acknowledgments

- Using Hugo Theme [https://github.com/imfing/hextra]
