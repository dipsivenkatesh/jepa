# Read the Docs configuration file
# See https://docs.readthedocs.io/en/stable/config-file/v2.html for details

version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.11"
  jobs:
    pre_create_environment: []
    post_checkout:
      - |
        if [ "$READTHEDOCS_VERSION_TYPE" = "external" ] && git diff --quiet origin/main HEAD -- docs/ .readthedocs.yaml;
        then
          echo "No changes to docs - canceling build"
          exit 183;
        fi

sphinx:
  configuration: docs/source/conf.py
  builder: html
  fail_on_warning: false

python:
  install:
    - requirements: docs/requirements.txt
    - method: pip
      path: .

formats:
  - pdf
  - epub
