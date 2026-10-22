while pgrep -f chain.sh >/dev/null; do sleep 5; done; python score_relevance.py > relev.log 2>&1
