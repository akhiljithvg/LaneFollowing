import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# GPIO motor pins (reversed to fix direction issue)
MOTOR_LEFT_FORWARD = 21
MOTOR_LEFT_BACKWARD = 20
MOTOR_RIGHT_FORWARD = 16
MOTOR_RIGHT_BACKWARD = 12

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Setup motor pins
motor_pins = [MOTOR_LEFT_FORWARD, MOTOR_LEFT_BACKWARD, MOTOR_RIGHT_FORWARD, MOTOR_RIGHT_BACKWARD]
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)

# Create PWM objects at 30 Hz
left_pwm = GPIO.PWM(MOTOR_LEFT_FORWARD, 30)
right_pwm = GPIO.PWM(MOTOR_RIGHT_FORWARD, 30)

left_pwm.start(0)
right_pwm.start(0)

def stop():
    """Stops both motors by setting their duty cycle to 0 and backward pins to LOW."""
    left_pwm.ChangeDutyCycle(0)
    right_pwm.ChangeDutyCycle(0)
    GPIO.output(MOTOR_LEFT_BACKWARD, GPIO.LOW)
    GPIO.output(MOTOR_RIGHT_BACKWARD, GPIO.LOW)

def move_forward(speed=10):
    """Moves the robot forward at a specified speed."""
    GPIO.output(MOTOR_LEFT_BACKWARD, GPIO.LOW)
    GPIO.output(MOTOR_RIGHT_BACKWARD, GPIO.LOW)
    left_pwm.ChangeDutyCycle(speed)
    right_pwm.ChangeDutyCycle(speed)

def turn_left(speed=10):
    """Turns the robot left by stopping the left motor and moving the right motor."""
    GPIO.output(MOTOR_LEFT_BACKWARD, GPIO.LOW)
    GPIO.output(MOTOR_RIGHT_BACKWARD, GPIO.LOW)
    left_pwm.ChangeDutyCycle(0)
    right_pwm.ChangeDutyCycle(speed)

def turn_right(speed=20):
    """Turns the robot right by stopping the right motor and moving the left motor."""
    GPIO.output(MOTOR_LEFT_BACKWARD, GPIO.LOW)
    GPIO.output(MOTOR_RIGHT_BACKWARD, GPIO.LOW)
    left_pwm.ChangeDutyCycle(speed)
    right_pwm.ChangeDutyCycle(0)

def pivot_left(speed=30):
    """Pivots the robot left by running left motor backward and right motor forward."""
    GPIO.output(MOTOR_LEFT_BACKWARD, GPIO.HIGH)
    GPIO.output(MOTOR_RIGHT_FORWARD, GPIO.HIGH)
    left_pwm.ChangeDutyCycle(speed)
    right_pwm.ChangeDutyCycle(speed)

def pivot_right(speed=30):
    """Pivots the robot right by running left motor forward and right motor backward."""
    GPIO.output(MOTOR_LEFT_FORWARD, GPIO.HIGH)
    GPIO.output(MOTOR_RIGHT_BACKWARD, GPIO.HIGH)
    left_pwm.ChangeDutyCycle(speed)
    right_pwm.ChangeDutyCycle(speed)

cap = cv2.VideoCapture(0)

FRAME_WIDTH = 320
CENTER_TOLERANCE = 20
PIVOT_THRESHOLD = 70  # Deviation after which pivoting is used

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, 240))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask = cv2.inRange(gray, 210, 255)

        roi = mask[160:240, :]

        contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                center = FRAME_WIDTH // 2
                cv2.circle(frame, (cx, 180), 5, (0, 0, 255), -1)
                print(f"Centroid at: {cx}")

                deviation = cx - center

                if deviation < -PIVOT_THRESHOLD:
                    print("Pivoting Right (sharp correction)")
                    pivot_left(speed=30)
                elif deviation > PIVOT_THRESHOLD:
                    print("Pivoting Left (sharp correction)")
                    pivot_right(speed=30)
                elif deviation < -CENTER_TOLERANCE:
                    print("Turning Right (mild correction)")
                    turn_left(speed=25)
                elif deviation > CENTER_TOLERANCE:
                    print("Turning Left (mild correction)")
                    turn_right(speed=25)
                else:
                    print("Aligned — Moving Forward")
                    move_forward(speed=9)
            else:
                print("No significant contour area found - Crawling ahead")
                move_forward(speed=9)
        else:
            print("Lane lost — Crawling ahead")
            move_forward(speed=9)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    print("Stopping motors and releasing resources.")
    stop()
    left_pwm.stop()
    right_pwm.stop()
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()