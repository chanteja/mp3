from flask import Flask, render_template, request, redirect, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import cv2
import os
import numpy as np
import threading

app = Flask(__name__)
app.secret_key = "college_ai_secret"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ================= DATABASE =================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))
    role = db.Column(db.String(20))

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    roll_no = db.Column(db.String(50))
    date = db.Column(db.String(20))
    in_time = db.Column(db.String(20))
    out_time = db.Column(db.String(20))

with app.app_context():
    db.create_all()

    if not User.query.filter_by(username="teacher1").first():
        db.session.add(User(
            username="teacher1",
            password=generate_password_hash("srivasavi"),
            role="teacher"
        ))

    if not User.query.filter_by(username="cst-301").first():
        db.session.add(User(
            username="cst-301",
            password=generate_password_hash("srivasavi"),
            role="class"
        ))

    db.session.commit()

# ================= GLOBAL SESSION =================

active_session = {}

# ================= FACE CAPTURE =================

def capture_faces(roll_no):
    path = os.path.join("dataset", roll_no)
    os.makedirs(path, exist_ok=True)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{path}/{count}.jpg", face)
            count += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        cv2.imshow("Capture Faces - Press Q", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 25:
            break

    cam.release()
    cv2.destroyAllWindows()

# ================= TRAIN MODEL =================

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_map = {}
    label_id = 0

    if not os.path.exists("dataset"):
        return None

    for student in os.listdir("dataset"):
        student_path = os.path.join("dataset", student)
        if not os.path.isdir(student_path):
            continue

        label_map[label_id] = student

        for img in os.listdir(student_path):
            gray = cv2.imread(os.path.join(student_path, img), cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                faces.append(gray)
                labels.append(label_id)

        label_id += 1

    if len(faces) == 0:
        return None

    recognizer.train(faces, np.array(labels))
    recognizer.save("trainer.yml")
    return label_map

# ================= MARK IN / OUT =================

def mark_in_time(roll_no):
    global active_session
    with app.app_context():
        today = datetime.now().strftime("%Y-%m-%d")
        record = Attendance.query.filter_by(
            roll_no=roll_no,
            date=today
        ).first()

        if not record:
            record = Attendance(
                roll_no=roll_no,
                date=today,
                in_time=datetime.now().strftime("%H:%M:%S"),
                out_time=None
            )
            db.session.add(record)
            db.session.commit()

    active_session[roll_no] = True

def mark_out_time():
    global active_session
    with app.app_context():
        today = datetime.now().strftime("%Y-%m-%d")

        for roll in active_session.keys():
            record = Attendance.query.filter_by(
                roll_no=roll,
                date=today
            ).first()

            if record:
                record.out_time = datetime.now().strftime("%H:%M:%S")

        db.session.commit()

    active_session.clear()

# ================= AI ATTENDANCE =================

def start_ai():
    global active_session

    label_map = train_model()
    if label_map is None:
        print("No training data found")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    print("AI Attendance Started")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            try:
                label, confidence = recognizer.predict(face)
            except:
                continue

            if confidence < 80:
                roll = label_map[label]

                if roll not in active_session:
                    mark_in_time(roll)

                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, roll,
                            (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0,255,0), 2)

        present_count = len(active_session)

        cv2.putText(frame,
                    f"Currently Present: {present_count}",
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,0,0),
                    2)

        cv2.putText(frame,
                    "Press Q to Stop",
                    (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,0,255),
                    2)

        cv2.imshow("AI Attendance - Live", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    mark_out_time()

# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login/<role>", methods=["GET","POST"])
def login(role):
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username, role=role).first()

        if user and check_password_hash(user.password, password):

            if role == "class":
                if not request.remote_addr.startswith("10.10."):
                    flash("Access denied. Not connected to class network.")
                    return redirect(f"/login/{role}")

            session["user"] = username
            session["role"] = role
            return redirect(f"/{role}_dashboard")

        flash("Invalid credentials")

    return render_template("login.html", role=role)

@app.route("/teacher_dashboard")
def teacher_dashboard():
    if session.get("role") != "teacher":
        return redirect("/")

    records = Attendance.query.all()
    processed = []

    for r in records:
        if r.out_time:
            in_time = datetime.strptime(r.in_time, "%H:%M:%S")
            out_time = datetime.strptime(r.out_time, "%H:%M:%S")
            minutes = (out_time - in_time).seconds / 60
            status = "Present" if minutes >= 40 else "Absent"
        else:
            minutes = 0
            status = "Absent"

        processed.append({
            "roll": r.roll_no,
            "date": r.date,
            "in_time": r.in_time,
            "out_time": r.out_time,
            "minutes": round(minutes,1),
            "status": status
        })

    return render_template("teacher_dashboard.html", records=processed)

@app.route("/student_dashboard")
def student_dashboard():
    if session.get("role") != "student":
        return redirect("/")

    roll = session.get("user")
    records = Attendance.query.filter_by(roll_no=roll).all()

    total = len(records)
    present_count = 0
    processed = []

    for r in records:
        if r.out_time:
            in_time = datetime.strptime(r.in_time, "%H:%M:%S")
            out_time = datetime.strptime(r.out_time, "%H:%M:%S")
            minutes = (out_time - in_time).seconds / 60
            status = "Present" if minutes >= 40 else "Absent"

            if status == "Present":
                present_count += 1
        else:
            minutes = 0
            status = "Absent"

        processed.append({
            "date": r.date,
            "in_time": r.in_time,
            "out_time": r.out_time,
            "minutes": round(minutes,1),
            "status": status
        })

    percentage = (present_count / total * 100) if total > 0 else 0

    return render_template("student_dashboard.html",
                           records=processed,
                           percentage=round(percentage,2))

@app.route("/class_dashboard")
def class_dashboard():
    if session.get("role") != "class":
        return redirect("/")
    return render_template("class_dashboard.html")

@app.route("/add_student", methods=["POST"])
def add_student():
    roll = request.form["roll"]

    if not User.query.filter_by(username=roll).first():
        db.session.add(User(
            username=roll,
            password=generate_password_hash("srivasavi"),
            role="student"
        ))
        db.session.commit()

    threading.Thread(target=capture_faces, args=(roll,)).start()
    return redirect("/class_dashboard")

@app.route("/start_ai_attendance", methods=["POST"])
def start_ai_attendance():
    threading.Thread(target=start_ai).start()
    return redirect("/class_dashboard")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ================= RUN =================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)