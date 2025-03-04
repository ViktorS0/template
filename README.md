# Template
project skeleton for python 

## Create ssh key to connect to git
in the console write `ssh-keygen -t ed25519` or `ssh-keygen -t RSA`. ed25519 is sorther more modern ssh key

The key generator should ask you to choose a location to store the keys. Press Enter to save them in the default location, which is the /.ssh directory in /home. Alternatively, specify a different location by typing its path and file name.

Next, the key generator window will ask you to create an SSH key passphrase to access your private key. You can press Enter to skip this step, but it is strongly encouraged to create a private key passphrase to enhance server connection security. The ssh-keygen command will now display the SSH key fingerprint and randomart image for the public key. The public key file should have a PUB extension, like id_rsa.pub or id_ed25519.pub.

To print the public key write `cat ~/.ssh/id_rsa.pub` or `cat ~/.ssh/id_ed25519.pub` or take it from the file directly 

Copy and paste the key in the git provider and save it.

set up user.name and user.email in the terminal 

`git config --global user.name "Your Name"`

`git config --global user.email "your.email@example.com"`

### Clone git repo 
to clone the git repository coppy the ssh link from git and write in the console `git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY`

## Create virtual environment 
In the console write
`python3 -m venv ~/.venv` or in git codespace `virtualenv ~/.venv`

Activate virtual environment 
`source ~/.venv/bin/activate`

Check environment
`which python`

To make sure this virtual env is activated when opren new consoles:
* open the bash file `vim ~/.bashrc`
* and paste `source ~/.venv/bin/activate` at the bottom of the file
* write the file and exit `:wq`

### Create virtual conda environment
1. make sure anaconda is instaled
2. cretate `environment.yml` file with the dependancies that need to be instaled
name: env-name
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.11
  - pandas
3. to activate the environment run `conda env create -f environment.yml`

