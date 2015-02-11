from gcm import GCM

API_KEY = "AIzaSyBA2JStV2Tv7YcyRFZcSYU-w4xvK_iC2Xg"
gcmVar = GCM(API_KEY)

def SendActivity(reg_id, activityRecognized):
	#print reg_id
	data = {'activity': activityRecognized}
	#print data
	#try:
	#	response = gcmVar.plaintext_request(registration_id=reg_id, data=data)
	#except Exception as inst:
	#	print inst.args
