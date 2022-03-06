# User specific environment and startup programs
module --force purge

export PATH=$PATH:$HOME/.local/bin:$HOME/bin

## set up CMSSW related code
alias scram="/cvmfs/cms.cern.ch/common/scram"
alias scramv1="/cvmfs/cms.cern.ch/common/scramv1"

source /cvmfs/cms.cern.ch/cmsset_default.sh

#export PATH=$PATH:/usr/local/cuda/bin

alias ll="ls -l"
