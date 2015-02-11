from celery import Celery
import activityRecognizer as acRecog
import time 

app = Celery('tasks', backend='amqp', broker='amqp://') 

# example task 1
@app.task(ignore_result=True)
def print_hello():
    print 'hello there'

# example task 2
@app.task
def gen_prime(x):
    multiples = []
    results = []
    for i in xrange(2, x+1):
        if i not in multiples:
            results.append(i)
            for j in xrange(i*i, x+1, i):
                multiples.append(j)
    return results

# Kick off recognition task
@app.task(ignore_result=True)
#@app.task
def startRecognition(collectionName):
    #cache.set(do_job.request.id, operation_results)
    #results = acRecog.startRecognition(collectionName, reg_id)
    results = acRecog.startRecognition(collectionName, reg_id)
    #return results
