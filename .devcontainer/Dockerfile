FROM nvcr.io/nvidia/tensorrt:26.05-py3

WORKDIR /workspace

 
 RUN cat <<'EOF' >> /root/.bashrc
 
 __bash_prompt() {
     local gitbranch='`\
         export BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || git rev-parse --short HEAD 2>/dev/null); \
         if [ "${BRANCH}" != "" ]; then \
             echo -n "\[\033[0;36m\](\[\033[1;31m\]${BRANCH}\[\033[0;36m\]) "; \
         fi`'
     local lightblue='\[\033[1;34m\]'
     local removecolor='\[\033[0m\]'
     PS1="${lightblue}\w ${gitbranch}${removecolor}\$ "
     unset -f __bash_prompt
 }
 __bash_prompt
EOF