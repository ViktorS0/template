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
Conda environment alows to manage more then just python packegies. It can also mange operating sysyem, python version and more

1. Instal miniconda if needed. Conda is available on Windows, macOS, or Linux and can be used with any terminal application
`conda update conda`
2. cretate `environment.yml` file with the dependancies that need to be instaled

```
name: env-name
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pandas
  - pip:
    - ollama
```

3. To activate the environment run
   1. run `conda info --envs` # to see a list of avalable environments 
   2. to create new envoronment from .yml file run `conda env create -f environment.yml` # to create conda environment form file environment.yml
   3. `conda activate <env-name>` # replace <env-name> with the name of the environment to be activated. The environment name can be found in the .yml file use in the setup.
### Update conda envoronment 
update conda env with environment.yml file. --prune removes packegies that are no longer in the file.

`conda env update --f environment.yml --prune` 

### Remove conda environment
`conda env remove --name ENV_NAME`

