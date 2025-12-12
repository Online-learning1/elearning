# main.py
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
import uuid

# Configuration
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(
    title="E-Learning Platform API",
    description="Modern e-learning platform backend",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Database models (in-memory for this example)
class User(BaseModel):
    id: str
    username: str
    email: EmailStr
    full_name: str
    disabled: bool = False
    enrolled_courses: List[str] = []
    role: str = "student"  # student, instructor, admin

class UserInDB(User):
    hashed_password: str

class Course(BaseModel):
    id: str
    title: str
    description: str
    instructor: str
    category: str
    duration_hours: int
    price: float
    rating: float = 0.0
    enrolled_students: int = 0
    image_url: str = ""
    published: bool = True
    created_at: datetime = datetime.now()

class Enrollment(BaseModel):
    id: str
    user_id: str
    course_id: str
    enrolled_at: datetime = datetime.now()
    progress: float = 0.0
    completed: bool = False

# In-memory storage (replace with database in production)
users_db = {}
courses_db = {}
enrollments_db = {}

# Pydantic models for requests/responses
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    full_name: str
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: EmailStr
    full_name: str
    role: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class CourseCreate(BaseModel):
    title: str
    description: str
    instructor: str
    category: str
    duration_hours: int
    price: float
    image_url: str = ""

# Utility functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(username: str):
    if username in users_db:
        user_dict = users_db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Initialize with sample data
@app.on_event("startup")
async def startup_event():
    # Create sample users
    sample_users = [
        {
            "id": str(uuid.uuid4()),
            "username": "student1",
            "email": "student1@example.com",
            "full_name": "John Student",
            "hashed_password": get_password_hash("password123"),
            "role": "student",
            "enrolled_courses": []
        },
        {
            "id": str(uuid.uuid4()),
            "username": "instructor1",
            "email": "instructor@example.com",
            "full_name": "Jane Instructor",
            "hashed_password": get_password_hash("password123"),
            "role": "instructor",
            "enrolled_courses": []
        }
    ]
    
    for user in sample_users:
        users_db[user["username"]] = user
    
    # Create sample courses
    sample_courses = [
        Course(
            id=str(uuid.uuid4()),
            title="Web Development Bootcamp",
            description="Learn modern web development with React, Node.js, and MongoDB",
            instructor="Jane Instructor",
            category="Development",
            duration_hours=40,
            price=99.99,
            rating=4.8,
            enrolled_students=1250,
            image_url="https://images.unsplash.com/photo-1555066931-4365d14bab8c?w=400&h=225&fit=crop"
        ),
        Course(
            id=str(uuid.uuid4()),
            title="Data Science Fundamentals",
            description="Master data analysis, visualization, and machine learning basics",
            instructor="Alex Johnson",
            category="Data Science",
            duration_hours=35,
            price=129.99,
            rating=4.9,
            enrolled_students=890,
            image_url="https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=400&h=225&fit=crop"
        ),
        Course(
            id=str(uuid.uuid4()),
            title="UI/UX Design Principles",
            description="Learn user-centered design for modern applications",
            instructor="Sarah Designer",
            category="Design",
            duration_hours=25,
            price=79.99,
            rating=4.7,
            enrolled_students=640,
            image_url="https://images.unsplash.com/photo-1561070791-2526d30994b5?w-400&h=225&fit=crop"
        ),
        Course(
            id=str(uuid.uuid4()),
            title="Python for Beginners",
            description="Start your programming journey with Python",
            instructor="Michael Python",
            category="Programming",
            duration_hours=30,
            price=49.99,
            rating=4.6,
            enrolled_students=2100,
            image_url="https://images.unsplash.com/photo-1526379879527-8559ecfcaec5?w=400&h=225&fit=crop"
        ),
        Course(
            id=str(uuid.uuid4()),
            title="Mobile App Development",
            description="Build iOS and Android apps with React Native",
            instructor="David Mobile",
            category="Mobile Development",
            duration_hours=45,
            price=149.99,
            rating=4.8,
            enrolled_students=720,
            image_url="https://images.unsplash.com/photo-1512941937669-90a1b58e7e9c?w=400&h=225&fit=crop"
        ),
        Course(
            id=str(uuid.uuid4()),
            title="Digital Marketing Mastery",
            description="Learn SEO, social media, and content marketing strategies",
            instructor="Lisa Marketer",
            category="Marketing",
            duration_hours=28,
            price=89.99,
            rating=4.5,
            enrolled_students=930,
            image_url="https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=400&h=225&fit=crop"
        )
    ]
    
    for course in sample_courses:
        courses_db[course.id] = course

# Auth endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register", response_model=UserResponse)
async def register_user(user_data: UserCreate):
    if user_data.username in users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    user_id = str(uuid.uuid4())
    hashed_password = get_password_hash(user_data.password)
    
    user = UserInDB(
        id=user_id,
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        hashed_password=hashed_password,
        role="student"
    )
    
    users_db[user.username] = user.dict()
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        role=user.role
    )

# User endpoints
@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

# Course endpoints
@app.get("/courses", response_model=List[Course])
async def get_courses(category: Optional[str] = None, search: Optional[str] = None):
    courses = list(courses_db.values())
    
    if category:
        courses = [c for c in courses if c.category == category]
    
    if search:
        search_lower = search.lower()
        courses = [c for c in courses if search_lower in c.title.lower() or search_lower in c.description.lower()]
    
    return courses

@app.get("/courses/{course_id}", response_model=Course)
async def get_course(course_id: str):
    if course_id not in courses_db:
        raise HTTPException(status_code=404, detail="Course not found")
    return courses_db[course_id]

@app.post("/courses", response_model=Course)
async def create_course(course_data: CourseCreate, current_user: User = Depends(get_current_active_user)):
    if current_user.role not in ["instructor", "admin"]:
        raise HTTPException(status_code=403, detail="Not authorized to create courses")
    
    course_id = str(uuid.uuid4())
    course = Course(
        id=course_id,
        **course_data.dict(),
        instructor=current_user.full_name,
        published=True
    )
    
    courses_db[course_id] = course
    return course

# Enrollment endpoints
@app.post("/enroll/{course_id}")
async def enroll_in_course(course_id: str, current_user: User = Depends(get_current_active_user)):
    if course_id not in courses_db:
        raise HTTPException(status_code=404, detail="Course not found")
    
    # Check if already enrolled
    for enrollment in enrollments_db.values():
        if enrollment.user_id == current_user.id and enrollment.course_id == course_id:
            raise HTTPException(status_code=400, detail="Already enrolled in this course")
    
    enrollment_id = str(uuid.uuid4())
    enrollment = Enrollment(
        id=enrollment_id,
        user_id=current_user.id,
        course_id=course_id
    )
    
    enrollments_db[enrollment_id] = enrollment
    
    # Update user's enrolled courses
    users_db[current_user.username]["enrolled_courses"].append(course_id)
    
    # Update course enrollment count
    course = courses_db[course_id]
    course.enrolled_students += 1
    
    return {"message": "Successfully enrolled in course", "enrollment_id": enrollment_id}

@app.get("/my-courses")
async def get_my_courses(current_user: User = Depends(get_current_active_user)):
    my_enrollments = [e for e in enrollments_db.values() if e.user_id == current_user.id]
    enrolled_courses = []
    
    for enrollment in my_enrollments:
        course = courses_db.get(enrollment.course_id)
        if course:
            enrolled_courses.append({
                "course": course,
                "enrollment": enrollment
            })
    
    return enrolled_courses

# Categories endpoint
@app.get("/categories")
async def get_categories():
    categories = set(course.category for course in courses_db.values())
    return list(categories)

# Stats endpoint
@app.get("/stats")
async def get_platform_stats():
    total_courses = len(courses_db)
    total_enrollments = len(enrollments_db)
    total_students = len([u for u in users_db.values() if u["role"] == "student"])
    
    return {
        "total_courses": total_courses,
        "total_enrollments": total_enrollments,
        "total_students": total_students,
        "categories": list(set(course.category for course in courses_db.values()))
    }

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to E-Learning Platform API", "version": "1.0.0"}
