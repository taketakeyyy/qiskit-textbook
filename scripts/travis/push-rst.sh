#!/bin/sh

setup_git() {
  git config --global user.email "travis@travis-ci.org"
  git config --global user.name "Travis CI"
}

commit_rst_files() {
  echo "Change branch to $1."
  git checkout "$1"
  echo "Add changes to *.rst files."
  git add documentation/\*.rst
  echo "Commit changes."
  git commit --message "Update RST files [skip ci]: $TRAVIS_BUILD_NUMBER"
}

upload_files() {
  echo "Push changes to $1."
  git remote add upstream https://${GITHUB_TOKEN}@github.com/qiskit/qiskit-textbook.git > /dev/null 2>&1
  git push upstream "$1"
}

setup_git
commit_rst_files $1
upload_files $1
