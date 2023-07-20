extends Spatial

var timer = Timer.new()
var calibrationtimer = Timer.new()
var gazeX = 0
var gazeY = 0
onready var target = $Viewport/Control/GazeTarget
onready var menumesh = $MenuMesh
onready var counterLabel = $Viewport/Control/ColorRect/Counter
onready var gazeViewport = $Camera/EyeTrackingCanvasMesh
onready var red_boxes = $Viewport/Control/ColorRect
var counter = 3
var cornerCounter = 6
var thread
var file

var client = StreamPeerTCP.new()
var server_address = "127.0.0.1"
var port = 8888
var error
var start = true
var message = "hello"

func _ready():
	file = File.new()
	file.open("res://Data/output.txt", File.WRITE)
	red_boxes.visible = false
	#target.visible = false
	#gazeViewport.visible = false
	pass

func _thread_function(userdata):
	# connect to the Python server
	error = client.connect_to_host(server_address, port)
	if error != OK:
		print("Error connecting to server: ", error)
	else:
		print("Connected to server on port: ", port)
	
	
	timer.connect("timeout",self,"_on_timer_timeout") 
	timer.set_wait_time(0.1)
	add_child(timer) #to process
	timer.start() #to start
	# send data to the server
	#var message = "Hello, Python!"
	#client.put_utf8_string(message)
	#client.flush()

	#while(client.get_status() == StreamPeerTCP.STATUS_CONNECTED):
	#	_receive_data()
	
	# close the connection
	#client.disconnect_from_host()

func _receive_data():
	# receive response from the server
	#if error != OK:
	#	print("Error connecting to server: ", error)
	#print(client.get_status())
	while client.get_status() == StreamPeerTCP.STATUS_CONNECTED:
		if client.get_available_bytes() > 0:
			var data = client.get_utf8_string(client.get_available_bytes())
			#print("Received:", data)
			var data_splitted = data.split(",")
			
			gazeX = float(data_splitted[0])
			gazeY = float(data_splitted[1])
			print("X: " + str(stepify(gazeX, 0.1)) + "   Y: " + str(stepify(gazeY, 0.1)) ) 
			#target.transform.origin.x = gazeX
			#target.transform.origin.y = gazeY
			
			#target.rect_position = Vector2(gazeX,gazeY)
			var timeDict = OS.get_time()
			var hour = timeDict.hour
			var minute = timeDict.minute
			var seconds = timeDict.second
			var content = str(hour) + ":" +str(minute) + ":" + str(seconds)+","+ str(int(gazeX)) + "," + str(int(gazeY)) + "\n"
			file.store_string(str(content))
			
			var tween:Tween = Tween.new()
			add_child(tween) #or in some cases call_deferred("add_child",tween)
			tween.interpolate_property(target, "rect_position",
			target.rect_position, Vector2(gazeX, gazeY), 0.2, Tween.TRANS_LINEAR, Tween.EASE_IN_OUT)
			tween.start()
			tween.connect("tween_all_completed",tween,"queue_free") #in case you call tween only one time, this line tells the tween node to auto destroy.
			#time = 0
			#for i in range(5):
				#time += 0.01
			#	target.rect_position = lerp(target.rect_position, Vector2(gazeX,gazeY), time)
			#	print("Lerp: ",lerp(target.rect_position, Vector2(gazeX,gazeY), 0.01))
		
		# send data to the server
		#message = "Hello, Python!"
		client.put_utf8_string(message)
		
		#client.flush()

		break
		#print("asd")
	

func _on_timer_timeout():
	if(start):
		_receive_data()

func _on_calibration_timer_timeout():
	if(counter>0):
		counter -= 1
		counterLabel.text = str(counter)
	else:
		counter = 3
		counterLabel.text = str(counter)
		cornerCounter -= 1
	
	if(cornerCounter==6):
		$Viewport/Control/ColorRect.rect_position = Vector2(0,0)
	if(cornerCounter==5):
		$Viewport/Control/ColorRect.rect_position = Vector2(1780,0)
	if(cornerCounter==4):
		$Viewport/Control/ColorRect.rect_position = Vector2(1780,880)
	if(cornerCounter==3):
		$Viewport/Control/ColorRect.rect_position = Vector2(0,880)
	if(cornerCounter==2):
		$Viewport/Control/ColorRect.rect_position = Vector2(395,440)
	if(cornerCounter==1):
		$Viewport/Control/ColorRect.rect_position = Vector2(1385,440)
	if(cornerCounter<1):
		#target.visible = true
		red_boxes.visible = false
		calibrationtimer.stop()
		#print("hello")
	
func _exit_tree():
	thread.wait_to_finish()
	file.close()

func _input(event):
	if Input.is_action_pressed("ui_accept"):
		thread = Thread.new()
		thread.start(self, "_thread_function", "param")
		#target.visible = true
		menumesh.visible = false
		gazeViewport.visible = true
		
	if Input.is_action_pressed("ui_cancel"):
		thread = Thread.new()
		thread.start(self, "_thread_function", "param")
		target.visible = true
		menumesh.visible = false
		gazeViewport.visible = true
		red_boxes.visible = false
	
	if Input.is_action_pressed("ui_down"):
		client.connect_to_host(server_address, port)
	if Input.is_action_pressed("ui_up"):
		client.disconnect_from_host()
	if Input.is_action_pressed("ui_right"):
		message = "reset_max_values"
	if Input.is_action_pressed("ui_left"):
		message = "continue"
	if Input.is_action_pressed("ui_end"):
		message = "stop_max_count"
	
	if Input.is_action_pressed("start_test"):
		calibrationtimer.connect("timeout",self,"_on_calibration_timer_timeout") 
		calibrationtimer.set_wait_time(1)
		add_child(calibrationtimer) #to process
		calibrationtimer.start()
		red_boxes.visible = true
		file.store_string("test_start\n")
		
func _notification(what):
	if what == MainLoop.NOTIFICATION_WM_QUIT_REQUEST:
		file.close()
