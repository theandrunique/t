import hashlib
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import punq
from fastapi import Depends, FastAPI, Request, params
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field, ValidationError
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class BaseORM(DeclarativeBase): ...


@dataclass
class EntityValidationException(Exception):
    errors: dict[str, str]

    @classmethod
    def from_pydantic_validation_error(cls, e: ValidationError) -> "EntityValidationException":
        return cls(errors={error["loc"][0]: error["msg"] for error in e.errors()})


@dataclass(kw_only=True)
class User:
    id: int | None = None
    username: str
    email: str
    hashed_password: bytes

    def validate(self) -> None:
        try:
            UserValidator(
                username=self.username,
                email=self.email,
                hashed_password=self.hashed_password,
            )
        except ValidationError as e:
            raise EntityValidationException.from_pydantic_validation_error(e)


class UserValidator(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    hashed_password: bytes


@dataclass(kw_only=True, frozen=True)
class CreateUserDTO:
    username: str
    email: str
    password: str


class UserORM(BaseORM):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True)
    email: Mapped[str] = mapped_column(unique=True)
    hashed_password: Mapped[bytes]

    @classmethod
    def from_entity(cls, user: User) -> "UserORM":
        return cls(
            username=user.username,
            email=user.email,
            hashed_password=user.hashed_password,
        )

    def to_entity(self) -> User:
        return User(
            id=self.id,
            username=self.username,
            email=self.email,
            hashed_password=self.hashed_password,
        )


class IUsersRepository(ABC):
    @abstractmethod
    async def add(self, user: User) -> User: ...
    @abstractmethod
    async def get_by_id(self, id: int) -> User | None: ...

    @abstractmethod
    async def get_by_email(self, email: str) -> User | None: ...

    @abstractmethod
    async def get_by_username(self, username: str) -> User | None: ...

    @abstractmethod
    async def get_by_username_or_email(self, username_or_email: str) -> User | None: ...


@dataclass
class SQLAlchemyUsersRepository(IUsersRepository):
    session_maker: async_sessionmaker[AsyncSession]

    async def add(self, user: User) -> User:
        async with self.session_maker() as session:
            user_model = UserORM(
                username=user.username,
                email=user.email,
                hashed_password=user.hashed_password,
            )
            session.add(user_model)
            await session.commit()
            await session.refresh(user_model, ("id",))
            user.id = user_model.id

        return user

    async def _get_one_or_none_from_query(self, query) -> User | None:
        async with self.session_maker() as session:
            result = await session.execute(query)

            user = result.scalars().one_or_none()
            if not user:
                return None

            return user.to_entity()

    async def get_by_id(self, id: int) -> User | None:
        return await self._get_one_or_none_from_query(select(UserORM).where(UserORM.id == id))

    async def get_by_email(self, email: str) -> User | None:
        return await self._get_one_or_none_from_query(select(UserORM).where(UserORM.email == email))

    async def get_by_username(self, username: str) -> User | None:
        return await self._get_one_or_none_from_query(select(UserORM).where(UserORM.username == username))

    async def get_by_username_or_email(self, username_or_email: str) -> User | None:
        return await self._get_one_or_none_from_query(
            select(UserORM).where(
                or_(
                    UserORM.username == username_or_email,
                    UserORM.email == username_or_email,
                )
            )
        )


class HashService:
    def hash(self, value: str) -> bytes:
        return hashlib.sha256(value.encode()).digest()

    def is_valid(self, value: str, hashed_value: bytes) -> bool:
        return hashed_value == self.hash(value)


@dataclass
class UsersService:
    repository: IUsersRepository
    hash_service: HashService

    async def create_new_user(self, user: User) -> User:
        return await self.repository.add(user)

    def convert_from_dto_to_entity(self, create_user_dto: CreateUserDTO) -> User:
        user = User(
            email=create_user_dto.email,
            username=create_user_dto.username,
            hashed_password=self.hash_service.hash(create_user_dto.password),
        )
        return user

    async def check_if_can_be_created(self, user: User) -> None:
        user.validate()

        existed_user = await self.repository.get_by_email(user.email)
        if existed_user:
            raise EmailAlreadyExists

        existed_user = await self.repository.get_by_username(user.username)
        if existed_user:
            raise UsernameAlreadyExists

    async def get_user_by_email(self, email: str) -> User | None:
        user = await self.repository.get_by_email(email)
        return user

    async def get_user_by_username(self, username: str) -> User | None:
        user = await self.repository.get_by_username(username)
        return user

    async def get_user_by_username_or_email(self, username_or_email: str) -> User | None:
        user = await self.repository.get_by_username_or_email(username_or_email)
        return user


class BaseAuthServiceException(Exception): ...


class UsernameAlreadyExists(BaseAuthServiceException): ...


class EmailAlreadyExists(BaseAuthServiceException): ...


class UserNotFound(BaseAuthServiceException): ...


class InvalidCredentials(BaseAuthServiceException): ...


@dataclass
class AuthService:
    users_service: UsersService
    hash_service: HashService

    async def register(self, create_user_dto: CreateUserDTO) -> User:
        user = self.users_service.convert_from_dto_to_entity(create_user_dto)

        await self.users_service.check_if_can_be_created(user)

        user = await self.users_service.create_new_user(user)
        return user

    async def authenticate(self, login: str, password: str) -> User:
        user = await self.users_service.get_user_by_username_or_email(login)
        if not user:
            raise UserNotFound

        if not self.hash_service.is_valid(password, user.hashed_password):
            raise InvalidCredentials

        return user


def get_container(request: Request) -> punq.Container:
    try:
        return request.app.state.container
    except AttributeError:
        raise AttributeError("Container not found, add container to the app state")


def Provide[T](dependency: type[T]) -> T:
    async def _dependency(container: punq.Container = Depends(get_container)) -> T:
        return container.resolve(dependency)  # type: ignore

    return params.Depends(_dependency)  # type: ignore


async def init_db() -> async_sessionmaker[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///database.db")

    async with engine.begin() as conn:
        await conn.run_sync(BaseORM.metadata.create_all)

    session_maker = async_sessionmaker[AsyncSession](engine)
    return session_maker


async def init_container() -> punq.Container:
    container = punq.Container()

    session_maker = await init_db()

    container.register(HashService, scope=punq.Scope.singleton)

    container.register(
        async_sessionmaker[AsyncSession],
        instance=session_maker,
        scope=punq.Scope.singleton,
    )

    container.register(UsersService, scope=punq.Scope.singleton)
    container.register(IUsersRepository, SQLAlchemyUsersRepository, scope=punq.Scope.singleton)

    container.register(AuthService, scope=punq.Scope.singleton)

    return container


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    container = await init_container()
    app.state.container = container
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(EntityValidationException)
async def entity_validation_exception_handler(request: Request, exc: EntityValidationException):
    return JSONResponse(
        status_code=422,
        content={"errors": exc.errors},
    )


@app.exception_handler(UsernameAlreadyExists)
async def username_already_exists_exception_handler(request: Request, exc: UsernameAlreadyExists):
    return JSONResponse(
        status_code=409,
        content={"detail": "Username already exists"},
    )


@app.exception_handler(EmailAlreadyExists)
async def email_already_exists_exception_handler(request: Request, exc: EmailAlreadyExists):
    return JSONResponse(
        status_code=409,
        content={"detail": "Email already exists"},
    )


@app.exception_handler(UserNotFound)
async def user_not_found_exception_handler(request: Request, exc: UserNotFound):
    return JSONResponse(
        status_code=404,
        content={"detail": "User not found"},
    )


@app.exception_handler(InvalidCredentials)
async def invalid_credentials_exception_handler(request: Request, exc: InvalidCredentials):
    return JSONResponse(
        status_code=401,
        content={"detail": "Invalid credentials"},
    )


class SignUpSchema(BaseModel):
    username: str
    email: str
    password: str


class SignInSchema(BaseModel):
    login: str
    password: str


class UserSchema(BaseModel):
    id: int
    email: str
    username: str


@app.post("/auth/sign-up", response_model=UserSchema)
async def sign_up(
    sign_up_schema: SignUpSchema,
    auth_service=Provide(AuthService),
):
    user = await auth_service.register(
        CreateUserDTO(
            username=sign_up_schema.username,
            email=sign_up_schema.email,
            password=sign_up_schema.password,
        )
    )

    return user


@app.post("/auth/sign-in", response_model=UserSchema)
async def sign_in(
    sign_in_schema: SignInSchema,
    auth_service=Provide(AuthService),
):
    user = await auth_service.authenticate(
        login=sign_in_schema.login,
        password=sign_in_schema.password,
    )

    return user
