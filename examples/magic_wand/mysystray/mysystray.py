from pystray import Icon , Menu, MenuItem
import os
import math
import numpy as np
from PIL import Image, ImageDraw
#from pyautogui import press, typewrite, hotkey
#from pyautogui import scroll, dragRel,dragTo,mouseDown,mouseUp
import pyautogui
import serial
import serial.tools.list_ports
import threading
import time
import re
import tensorflow as tf
#import subprocess
import tkinter as tk


serialPort = serial.Serial(
    #port="COM32", baudrate=9600, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
    port="COM32", baudrate=9600
)


serialString = ""  # Used to hold data coming over UART
active = False
GMapState = False
ChromeState = False
apprun = True

#==============================tensorflow, inference================================
FIXED_POINT = 4096
LABEL_MAPPING=dict(enumerate(["down", "left", "ok", "right", "up"]))
QUANTIZED_TFL_MODEL_FILENAME = "quantized_model.tfl"
QUANTIZED_TFL_MODEL_PATH=os.path.join(os.getcwd(), QUANTIZED_TFL_MODEL_FILENAME)

def mul_fp(a, b):
  return (a * b) // FIXED_POINT

def div_fp(a, b):
  if b == 0:
    b = 1
  return (a * FIXED_POINT) // b

def float_to_fp(a):
  return math.floor(a * FIXED_POINT)

def norm_to_coord_fp(a, range_fp, half_size_fp):
  a_fp = float_to_fp(a)
  norm_fp = div_fp(a_fp, range_fp)
  return mul_fp(norm_fp, half_size_fp) + half_size_fp

# 四捨五入 (rounding)
def round_fp_to_int(a):
  return math.floor((a + (FIXED_POINT / 2)) / FIXED_POINT)

def gate(a, min, max):
  if a < min:
    return min
  elif a > max:
    return max
  else:
    return a

def rasterize_stroke(stroke_points, x_range, y_range, width, height):
  num_channels = 3
  buffer_byte_count = height * width * num_channels
  buffer = bytearray(buffer_byte_count)

  width_fp = width * FIXED_POINT
  height_fp = height * FIXED_POINT
  half_width_fp = width_fp / 2
  half_height_fp = height_fp / 2
  x_range_fp = float_to_fp(x_range)
  y_range_fp = float_to_fp(y_range)

  t_inc_fp = FIXED_POINT // len(stroke_points)

  one_half_fp = (FIXED_POINT / 2)

  for point_index in range(len(stroke_points) - 1):
    start_point = stroke_points[point_index]
    end_point = stroke_points[point_index + 1]
    start_x_fp = norm_to_coord_fp(start_point["x"], x_range_fp, half_width_fp)
    start_y_fp = norm_to_coord_fp(-start_point["y"], y_range_fp, half_height_fp)
    end_x_fp = norm_to_coord_fp(end_point["x"], x_range_fp, half_width_fp)
    end_y_fp = norm_to_coord_fp(-end_point["y"], y_range_fp, half_height_fp)
    delta_x_fp = end_x_fp - start_x_fp
    delta_y_fp = end_y_fp - start_y_fp

    # This just want to add a Gradient color
    # red->green->blue
    t_fp = point_index * t_inc_fp
    if t_fp < one_half_fp:
      local_t_fp = div_fp(t_fp, one_half_fp)
      one_minus_t_fp = FIXED_POINT - local_t_fp
      red = round_fp_to_int(one_minus_t_fp * 255)
      green = round_fp_to_int(local_t_fp * 255)
      blue = 0
    else:
      local_t_fp = div_fp(t_fp - one_half_fp, one_half_fp)
      one_minus_t_fp = FIXED_POINT - local_t_fp
      red = 0
      green = round_fp_to_int(one_minus_t_fp * 255)
      blue = round_fp_to_int(local_t_fp * 255)
    red = gate(red, 0, 255)
    green = gate(green, 0, 255)
    blue = gate(blue, 0, 255)

    #  mainly horizontal
    if abs(delta_x_fp) > abs(delta_y_fp):
      line_length = abs(round_fp_to_int(delta_x_fp))
      if delta_x_fp > 0:
        x_inc_fp = 1 * FIXED_POINT
        y_inc_fp = div_fp(delta_y_fp, delta_x_fp)
      else:
        x_inc_fp = -1 * FIXED_POINT
        y_inc_fp = -div_fp(delta_y_fp, delta_x_fp)
    #  mainly vertical
    else:
      line_length = abs(round_fp_to_int(delta_y_fp))
      if delta_y_fp > 0:
        y_inc_fp = 1 * FIXED_POINT
        x_inc_fp = div_fp(delta_x_fp, delta_y_fp)
      else:
        y_inc_fp = -1 * FIXED_POINT
        x_inc_fp = -div_fp(delta_x_fp, delta_y_fp)
    for i in range(line_length + 1):
      x_fp = start_x_fp + (i * x_inc_fp)
      y_fp = start_y_fp + (i * y_inc_fp)
      x = round_fp_to_int(x_fp)
      y = round_fp_to_int(y_fp)
      if (x < 0) or (x >= width) or (y < 0) or (y >= height):
        continue
      buffer_index = (y * width * num_channels) + (x * num_channels)
      buffer[buffer_index + 0] = red
      buffer[buffer_index + 1] = green
      buffer[buffer_index + 2] = blue
  
  np_buffer = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, num_channels)

  return np_buffer

def predict_tflite(tflite_model_path, img_array):
  # img = keras.preprocessing.image.load_img(filename, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
  # img_array = keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)

  # Initialize the TFLite interpreter
  interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  # If required, quantize the input layer (from float to integer)
  input_scale, input_zero_point = input_details["quantization"]
  if (input_scale, input_zero_point) != (0.0, 0):
    img_array = np.multiply(img_array, 1.0 / input_scale) + input_zero_point
    img_array = img_array.astype(input_details["dtype"])
  
  # Invoke the interpreter
  interpreter.set_tensor(input_details["index"], img_array)
  interpreter.invoke()
  pred = interpreter.get_tensor(output_details["index"])[0]
  
  # If required, dequantized the output layer (from integer to float)
  output_scale, output_zero_point = output_details["quantization"]
  if (output_scale, output_zero_point) != (0.0, 0):
    pred = pred.astype(np.float32)
    pred = np.multiply((pred - output_zero_point), output_scale)
  
  predicted_label_index = np.argmax(pred)
  predicted_score = pred[predicted_label_index]
  return (predicted_label_index, predicted_score)
#========================================================================
#========================================================================


#=============COM port reading===============================    
def comRead():
    global apprun
    w,h=pyautogui.size()
    print(w,h)
    cX = w/2
    cY = h/2
    while 1:
        if apprun == False:
            print("exit thread!")
            break
        # Wait until there is data waiting in the serial buffer
        if serialPort.in_waiting > 0:# 若收到序列資料…
            # Read data out of the buffer until a carraige return / new line is found
            data_raw = serialPort.readline()
            data = data_raw.decode() # 用預設的UTF-8解碼
            data2=data.strip()
            data2=re.split('[,\s]', data2)
            if 'AAAAA:' == data2[0]:
                print(data2[1])
                print(data2[2])
                one_strokes = [int(i) for i in data2[3:]]
                one_strokes = np.array(one_strokes) / 128
                strokePoints=[]
                for i in range(len(one_strokes)//2):
                    strokePoints.append({'x':one_strokes[2*i], 
                                         'y':one_strokes[2*i+1]
                                        })

                # print(one_strokes)
                # plot_stroke(strokePoints)
                
                raster = rasterize_stroke(strokePoints, 0.5, 0.5, 32, 32)
                img = Image.fromarray(raster).resize((128, 128), Image.NEAREST)

                #img.show()
                #print(QUANTIZED_TFL_MODEL_PATH)
                start_time = time.time()
                index, score = predict_tflite(QUANTIZED_TFL_MODEL_PATH, raster)
                end_time = time.time()
                print(f"predict:{LABEL_MAPPING[index]}({index}), score:{score*100*100//100}, cost:{(end_time-start_time)*1000*100//100}ms")
                #===============autoui===============
                try:
                    if LABEL_MAPPING[index] == "down":
                        print("down")
                        pyautogui.scroll(-200)  #scroll down / google map zoom out
                    elif LABEL_MAPPING[index] == "up":
                        print("up")
                        pyautogui.scroll(200)  #scroll up / google map zoom in
                    elif LABEL_MAPPING[index] == "right":
                        print("move right")
                        pyautogui.moveTo(cX,cY)
                        pyautogui.dragRel(100, 0, duration=2, button='left') #move right
                    elif LABEL_MAPPING[index] == "left":
                        print("move left")
                        pyautogui.moveTo(cX,cY)
                        pyautogui.dragRel(-100, 0, duration=2, button='left') #move left
                except:
                    pass
                #====================================
            else:
                # print(data)
                pass
#=======================================================================

#=======main, system tray===============================================
def checkArduinoPort():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        print(p)


def showOSD(notificationTimeout, textToDisplay):

    ## Create main window
    root = tk.Tk()
    tk.Label(root,text=textToDisplay, font=('Times','30'), fg='yellow', bg='white').pack(side=tk.RIGHT)
    root.wm_attributes("-topmost", 1)
    root.wm_attributes("-transparentcolor", "white")
    root.geometry('300x150+1000+100')
    root.update_idletasks()
    # Remove window decorations
    root.overrideredirect(1)
    timeOut = int(notificationTimeout*1000) # Convert to ms from s
    ## Run OSD appliction
    root.after(timeOut,root.destroy) # timeout to destroy OSD application
    root.mainloop()

def GMap(icon, item): #load google map action profile
    global ChromeState
    global GMapState
    GMapState = not item.checked
    ChromeState = False
    print("GoogleMap> " , GMapState)

def Chrome(icon, item): #load chrome action profile
    global ChromeState
    global GMapState
    ChromeState = not item.checked
    GMapState = False
    print("Chrome> " , ChromeState)

def exitProgram(icon):
    global apprun
    apprun = False
    icon.stop()

image = Image.open('test3.ico')
image.load()

menu = Menu(MenuItem('GoogleMap', GMap,checked=lambda item: GMapState),
            MenuItem('Chrome', Chrome,checked=lambda item: ChromeState),
            MenuItem('Exit', exitProgram))

sysicon = Icon('test name',image,"title", menu)


#checkArduinoPort()

#================a thread for COM reading================
#need to create a thread for reading data from COM port
t = threading.Thread(target = comRead)
t.start()
#======================================================

#showOSD(2,"Test")  #OSD testing

sysicon.run()

t.join()

print("Exit app!!")
#===============================================================================