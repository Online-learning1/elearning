# main.py
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, ConfigDict
from passlib.context import CryptContext
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Boolean, Integer, Float, DateTime, Text, ForeignKey, Table, UniqueConstraint
from sqlalchemy.orm import relationship, mapped_column, Mapped
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
import asyncpg

# Database configuration
DATABASE_URL = "postgresql+asyncpg://kabz_styles_user:qI0rZrBBYfwEP0RvIbblNd1Nd5bhz7bB@dpg-d4u2nqogjchc7397fvq0-a/kabz_styles"

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

# Configuration
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(
    title="E-Learning Platform API",
    description="Modern e-learning platform backend with PostgreSQL",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    full_name = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    disabled = Column(Boolean, default=False)
    role = Column(String(20), default="student")  # student, instructor, admin
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    enrollments = relationship("Enrollment", back_populates="user", cascade="all, delete-orphan")
    created_courses = relationship("Course", back_populates="instructor_user", foreign_keys="Course.instructor_id")

# Association table for course categories (many-to-many)
course_categories = Table(
    'course_categories',
    Base.metadata,
    Column('course_id', UUID(as_uuid=True), ForeignKey('courses.id')),
    Column('category_id', UUID(as_uuid=True), ForeignKey('categories.id'))
)

class Category(Base):
    __tablename__ = "categories"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    courses = relationship("Course", secondary=course_categories, back_populates="categories")

class Course(Base):
    __tablename__ = "courses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=False)
    instructor_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    duration_hours = Column(Integer, default=0)
    price = Column(Float, default=0.0)
    rating = Column(Float, default=0.0)
    enrolled_students = Column(Integer, default=0)
    image_url = Column(String(500))
    published = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    instructor_user = relationship("User", back_populates="created_courses", foreign_keys=[instructor_id])
    enrollments = relationship("Enrollment", back_populates="course", cascade="all, delete-orphan")
    categories = relationship("Category", secondary=course_categories, back_populates="courses")
    modules = relationship("CourseModule", back_populates="course", cascade="all, delete-orphan")

class CourseModule(Base):
    __tablename__ = "course_modules"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    course_id = Column(UUID(as_uuid=True), ForeignKey("courses.id"), nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    order = Column(Integer, default=0)
    duration_minutes = Column(Integer, default=0)
    video_url = Column(String(500))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    course = relationship("Course", back_populates="modules")
    progress = relationship("UserProgress", back_populates="module", cascade="all, delete-orphan")

class Enrollment(Base):
    __tablename__ = "enrollments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    course_id = Column(UUID(as_uuid=True), ForeignKey("courses.id"), nullable=False)
    enrolled_at = Column(DateTime(timezone=True), server_default=func.now())
    progress = Column(Float, default=0.0)
    completed = Column(Boolean, default=False)
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("User", back_populates="enrollments")
    course = relationship("Course", back_populates="enrollments")
    progress_records = relationship("UserProgress", back_populates="enrollment", cascade="all, delete-orphan")
    
    # Unique constraint
    __table_args__ = (UniqueConstraint('user_id', 'course_id', name='unique_user_course'),)

class UserProgress(Base):
    __tablename__ = "user_progress"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    enrollment_id = Column(UUID(as_uuid=True), ForeignKey("enrollments.id"), nullable=False)
    module_id = Column(UUID(as_uuid=True), ForeignKey("course_modules.id"), nullable=False)
    completed = Column(Boolean, default=False)
    completed_at = Column(DateTime(timezone=True))
    watched_duration = Column(Integer, default=0)  # in seconds
    last_position = Column(Integer, default=0)  # last watched position in video
    
    # Relationships
    enrollment = relationship("Enrollment", back_populates="progress_records")
    module = relationship("CourseModule", back_populates="progress")

# Pydantic models for requests/responses
class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: str

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: uuid.UUID
    role: str
    disabled: bool
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class CategoryBase(BaseModel):
    name: str
    description: Optional[str] = None

class CategoryResponse(CategoryBase):
    id: uuid.UUID
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class CourseBase(BaseModel):
    title: str
    description: str
    duration_hours: int
    price: float
    image_url: Optional[str] = None
    published: bool = True

class CourseCreate(CourseBase):
    category_ids: Optional[List[uuid.UUID]] = []

class CourseResponse(CourseBase):
    id: uuid.UUID
    instructor_id: uuid.UUID
    instructor_name: str
    rating: float
    enrolled_students: int
    categories: List[CategoryResponse]
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)

class EnrollmentResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    course_id: uuid.UUID
    enrolled_at: datetime
    progress: float
    completed: bool
    completed_at: Optional[datetime] = None
    course: CourseResponse
    
    model_config = ConfigDict(from_attributes=True)

# Utility functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Database dependency
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def get_user_by_username(session: AsyncSession, username: str) -> Optional[User]:
    result = await session.execute(
        select(User).where(User.username == username)
    )
    return result.scalar_one_or_none()

async def get_user_by_email(session: AsyncSession, email: str) -> Optional[User]:
    result = await session.execute(
        select(User).where(User.email == email)
    )
    return result.scalar_one_or_none()

async def authenticate_user(session: AsyncSession, username: str, password: str) -> Optional[User]:
    user = await get_user_by_username(session, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_db)
) -> User:
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
    
    user = await get_user_by_username(session, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Database initialization
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create default categories
    async with AsyncSessionLocal() as session:
        # Check if categories already exist
        result = await session.execute(select(Category))
        existing_categories = result.scalars().all()
        
        if not existing_categories:
            default_categories = [
                Category(name="Development", description="Web and software development"),
                Category(name="Data Science", description="Data analysis and machine learning"),
                Category(name="Design", description="UI/UX and graphic design"),
                Category(name="Business", description="Business and entrepreneurship"),
                Category(name="Marketing", description="Digital marketing and SEO"),
                Category(name="Personal Development", description="Soft skills and productivity"),
            ]
            
            session.add_all(default_categories)
            await session.commit()
        
        # Create admin user if not exists
        admin_user = await get_user_by_username(session, "admin")
        if not admin_user:
            admin_user = User(
                username="admin",
                email="admin@elearning.com",
                full_name="Administrator",
                hashed_password=get_password_hash("admin123"),
                role="admin"
            )
            session.add(admin_user)
            await session.commit()

# Add startup event
@app.on_event("startup")
async def on_startup():
    await init_db()

# Auth endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_db)
):
    user = await authenticate_user(session, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role}, 
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserCreate,
    session: AsyncSession = Depends(get_db)
):
    # Check if username exists
    existing_user = await get_user_by_username(session, user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email exists
    existing_email = await get_user_by_email(session, user_data.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    user = User(
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        hashed_password=hashed_password,
        role="student"
    )
    
    session.add(user)
    await session.commit()
    await session.refresh(user)
    
    return user

# User endpoints
@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

# Category endpoints
@app.get("/categories", response_model=List[CategoryResponse])
async def get_categories(session: AsyncSession = Depends(get_db)):
    result = await session.execute(select(Category).order_by(Category.name))
    categories = result.scalars().all()
    return categories

@app.post("/categories", response_model=CategoryResponse)
async def create_category(
    category_data: CategoryBase,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to create categories")
    
    category = Category(**category_data.dict())
    session.add(category)
    await session.commit()
    await session.refresh(category)
    
    return category

# Course endpoints
@app.get("/courses", response_model=List[CourseResponse])
async def get_courses(
    category_id: Optional[uuid.UUID] = None,
    search: Optional[str] = None,
    instructor_id: Optional[uuid.UUID] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    session: AsyncSession = Depends(get_db)
):
    query = select(Course).where(Course.published == True)
    
    if category_id:
        query = query.join(Course.categories).where(Category.id == category_id)
    
    if search:
        search_term = f"%{search}%"
        query = query.where(
            or_(
                Course.title.ilike(search_term),
                Course.description.ilike(search_term)
            )
        )
    
    if instructor_id:
        query = query.where(Course.instructor_id == instructor_id)
    
    if min_price is not None:
        query = query.where(Course.price >= min_price)
    
    if max_price is not None:
        query = query.where(Course.price <= max_price)
    
    query = query.order_by(Course.created_at.desc())
    result = await session.execute(query)
    courses = result.scalars().all()
    
    return courses

@app.get("/courses/{course_id}", response_model=CourseResponse)
async def get_course(course_id: uuid.UUID, session: AsyncSession = Depends(get_db)):
    result = await session.execute(
        select(Course).where(Course.id == course_id)
    )
    course = result.scalar_one_or_none()
    
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    return course

@app.post("/courses", response_model=CourseResponse)
async def create_course(
    course_data: CourseCreate,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    if current_user.role not in ["instructor", "admin"]:
        raise HTTPException(status_code=403, detail="Not authorized to create courses")
    
    # Create course
    course_dict = course_data.dict(exclude={"category_ids"})
    course = Course(
        **course_dict,
        instructor_id=current_user.id,
        instructor_name=current_user.full_name
    )
    
    # Add categories
    if course_data.category_ids:
        result = await session.execute(
            select(Category).where(Category.id.in_(course_data.category_ids))
        )
        categories = result.scalars().all()
        course.categories = categories
    
    session.add(course)
    await session.commit()
    await session.refresh(course)
    
    return course

@app.put("/courses/{course_id}", response_model=CourseResponse)
async def update_course(
    course_id: uuid.UUID,
    course_data: CourseCreate,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    result = await session.execute(
        select(Course).where(Course.id == course_id)
    )
    course = result.scalar_one_or_none()
    
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    if current_user.role != "admin" and course.instructor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to update this course")
    
    # Update course fields
    for field, value in course_data.dict(exclude={"category_ids"}).items():
        setattr(course, field, value)
    
    # Update categories if provided
    if course_data.category_ids is not None:
        result = await session.execute(
            select(Category).where(Category.id.in_(course_data.category_ids))
        )
        categories = result.scalars().all()
        course.categories = categories
    
    course.updated_at = datetime.utcnow()
    await session.commit()
    await session.refresh(course)
    
    return course

# Enrollment endpoints
@app.post("/enroll/{course_id}")
async def enroll_in_course(
    course_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    # Check if course exists
    result = await session.execute(
        select(Course).where(Course.id == course_id)
    )
    course = result.scalar_one_or_none()
    
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    # Check if already enrolled
    result = await session.execute(
        select(Enrollment).where(
            Enrollment.user_id == current_user.id,
            Enrollment.course_id == course_id
        )
    )
    existing_enrollment = result.scalar_one_or_none()
    
    if existing_enrollment:
        raise HTTPException(status_code=400, detail="Already enrolled in this course")
    
    # Create enrollment
    enrollment = Enrollment(
        user_id=current_user.id,
        course_id=course_id
    )
    
    # Update course enrollment count
    course.enrolled_students += 1
    
    session.add(enrollment)
    await session.commit()
    await session.refresh(enrollment)
    
    return {"message": "Successfully enrolled in course", "enrollment_id": enrollment.id}

@app.get("/my-courses", response_model=List[EnrollmentResponse])
async def get_my_courses(
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    result = await session.execute(
        select(Enrollment)
        .where(Enrollment.user_id == current_user.id)
        .order_by(Enrollment.enrolled_at.desc())
    )
    enrollments = result.scalars().all()
    
    return enrollments

@app.put("/enrollments/{enrollment_id}/progress")
async def update_enrollment_progress(
    enrollment_id: uuid.UUID,
    progress: float = Body(..., ge=0, le=100),
    completed: Optional[bool] = None,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    result = await session.execute(
        select(Enrollment).where(
            Enrollment.id == enrollment_id,
            Enrollment.user_id == current_user.id
        )
    )
    enrollment = result.scalar_one_or_none()
    
    if not enrollment:
        raise HTTPException(status_code=404, detail="Enrollment not found")
    
    enrollment.progress = progress
    
    if completed is not None and completed and not enrollment.completed:
        enrollment.completed = True
        enrollment.completed_at = datetime.utcnow()
    
    await session.commit()
    
    return {"message": "Progress updated successfully", "progress": progress}

# Course modules endpoints
@app.get("/courses/{course_id}/modules")
async def get_course_modules(
    course_id: uuid.UUID,
    session: AsyncSession = Depends(get_db)
):
    result = await session.execute(
        select(CourseModule)
        .where(CourseModule.course_id == course_id)
        .order_by(CourseModule.order)
    )
    modules = result.scalars().all()
    return modules

@app.post("/courses/{course_id}/modules")
async def create_course_module(
    course_id: uuid.UUID,
    module_data: dict,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    # Check if course exists and user is instructor
    result = await session.execute(
        select(Course).where(Course.id == course_id)
    )
    course = result.scalar_one_or_none()
    
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    if current_user.role != "admin" and course.instructor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to add modules to this course")
    
    module = CourseModule(
        course_id=course_id,
        **module_data
    )
    
    session.add(module)
    await session.commit()
    await session.refresh(module)
    
    return module

# User progress endpoints
@app.post("/progress")
async def update_user_progress(
    progress_data: dict,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    enrollment_id = progress_data.get("enrollment_id")
    module_id = progress_data.get("module_id")
    completed = progress_data.get("completed", False)
    watched_duration = progress_data.get("watched_duration", 0)
    last_position = progress_data.get("last_position", 0)
    
    # Check if enrollment exists and belongs to user
    result = await session.execute(
        select(Enrollment).where(
            Enrollment.id == enrollment_id,
            Enrollment.user_id == current_user.id
        )
    )
    enrollment = result.scalar_one_or_none()
    
    if not enrollment:
        raise HTTPException(status_code=404, detail="Enrollment not found")
    
    # Check if progress record exists
    result = await session.execute(
        select(UserProgress).where(
            UserProgress.enrollment_id == enrollment_id,
            UserProgress.module_id == module_id
        )
    )
    progress = result.scalar_one_or_none()
    
    if progress:
        progress.completed = completed
        progress.watched_duration = watched_duration
        progress.last_position = last_position
        if completed and not progress.completed_at:
            progress.completed_at = datetime.utcnow()
    else:
        progress = UserProgress(
            enrollment_id=enrollment_id,
            module_id=module_id,
            completed=completed,
            watched_duration=watched_duration,
            last_position=last_position,
            completed_at=datetime.utcnow() if completed else None
        )
        session.add(progress)
    
    await session.commit()
    
    # Update overall enrollment progress
    result = await session.execute(
        select(func.count(CourseModule.id))
        .select_from(CourseModule)
        .where(CourseModule.course_id == enrollment.course_id)
    )
    total_modules = result.scalar()
    
    result = await session.execute(
        select(func.count(UserProgress.id))
        .select_from(UserProgress)
        .join(Enrollment)
        .where(
            Enrollment.id == enrollment_id,
            UserProgress.completed == True
        )
    )
    completed_modules = result.scalar()
    
    if total_modules > 0:
        enrollment.progress = (completed_modules / total_modules) * 100
        if enrollment.progress >= 100 and not enrollment.completed:
            enrollment.completed = True
            enrollment.completed_at = datetime.utcnow()
    
    await session.commit()
    
    return {"message": "Progress updated successfully"}

# Stats endpoint
@app.get("/stats")
async def get_platform_stats(session: AsyncSession = Depends(get_db)):
    # Total courses
    result = await session.execute(select(func.count(Course.id)))
    total_courses = result.scalar()
    
    # Total enrollments
    result = await session.execute(select(func.count(Enrollment.id)))
    total_enrollments = result.scalar()
    
    # Total students
    result = await session.execute(select(func.count(User.id)).where(User.role == "student"))
    total_students = result.scalar()
    
    # Total instructors
    result = await session.execute(select(func.count(User.id)).where(User.role == "instructor"))
    total_instructors = result.scalar()
    
    # Popular categories
    result = await session.execute(
        select(Category.name, func.count(Course.id).label("course_count"))
        .select_from(Category)
        .join(Category.courses)
        .group_by(Category.id)
        .order_by(func.count(Course.id).desc())
        .limit(5)
    )
    popular_categories = result.all()
    
    # Recent courses
    result = await session.execute(
        select(Course)
        .order_by(Course.created_at.desc())
        .limit(5)
    )
    recent_courses = result.scalars().all()
    
    return {
        "total_courses": total_courses,
        "total_enrollments": total_enrollments,
        "total_students": total_students,
        "total_instructors": total_instructors,
        "popular_categories": [
            {"name": cat[0], "course_count": cat[1]} for cat in popular_categories
        ],
        "recent_courses": recent_courses
    }

# Search endpoint
@app.get("/search")
async def search_courses(
    q: str,
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    session: AsyncSession = Depends(get_db)
):
    query = select(Course).where(Course.published == True)
    
    if q:
        search_term = f"%{q}%"
        query = query.where(
            or_(
                Course.title.ilike(search_term),
                Course.description.ilike(search_term)
            )
        )
    
    if category:
        query = query.join(Course.categories).where(Category.name == category)
    
    if min_price is not None:
        query = query.where(Course.price >= min_price)
    
    if max_price is not None:
        query = query.where(Course.price <= max_price)
    
    query = query.order_by(Course.rating.desc())
    result = await session.execute(query)
    courses = result.scalars().all()
    
    return courses

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to E-Learning Platform API",
        "version": "2.0.0",
        "docs": "/docs",
        "database": "PostgreSQL"
    }
