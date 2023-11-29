import turtle
import time
import random
import threading as tr
from AIAgent import agent


class Snake:
    def __init__(self):
        self.delay = 0.1
        self.score = 0
        self.high_score = 0
        self.game_size = 600
        self.done = False
        self.bodies = []

        wn = turtle.Screen()
        wn.title("Snake Game")
        wn.bgcolor("gray")
        wn.setup(width=self.game_size, height=self.game_size)
        wn.tracer(0)

        # wn.listen()
        # wn.onkeypress(lambda: self.set_direction([1, 0, 0, 0]), "w")
        # wn.onkeypress(lambda: self.set_direction([0, 0, 1, 0]), "s")
        # wn.onkeypress(lambda: self.set_direction([0, 0, 0, 1]), "a")
        # wn.onkeypress(lambda: self.set_direction([0, 1, 0, 0]), "d")

        self.wn = wn
        self.put_head()
        self.put_food()

    def colision_occurred(self):
        for body in self.bodies:
            if self.head.distance(body) < 20:
                return True

        if self.head.xcor() <= -self.game_size/2 or self.head.xcor() >= self.game_size/2 or self.head.ycor() <= -self.game_size/2 or self.head.ycor() >= self.game_size/2:
            return True
        else:
            return False

    def food_caught(self):
        if self.head.distance(self.food) < 25:
            body = turtle.Turtle()
            body.shape("square")
            if len(self.bodies) % 2 == 0:
                body.color((0.4, 0.4, 0.4))
            else:
                body.color((0.1, 0.1, 0.1))

            body.speed(0)
            body.penup()
            body.goto(self.head.xcor(), self.head.ycor())
            if self.direction == [1, 0, 0, 0]:
                self.head.goto(self.head.xcor(), self.head.ycor()+20)
            elif self.direction == [0, 1, 0, 0]:
                self.head.goto(self.head.xcor()+20, self.head.ycor())
            elif self.direction == [0, 0, 1, 0]:
                self.head.goto(self.head.xcor(), self.head.ycor()-20)
            elif self.direction == [0, 0, 1, 0]:
                self.head.goto(self.head.xcor()-20, self.head.ycor())

            self.bodies.insert(0, body)

            self.score += 1
            if self.score > self.high_score:
                self.high_score = self.score
            return True
        else:
            return False

    def game_over(self):
        self.score = 0
        self.done = True
        self.restart_game()

    def restart_game(self):
        self.done = False
        self.head.goto(0, 0)
        self.direction = [1, 0, 0, 0]
        self.put_food()

        for i in self.bodies:
            i.hideturtle()

        del self.bodies
        self.bodies = []

    def next_step(self):
        if self.colision_occurred():
            self.game_over()
            return False
        else:
            if self.food_caught():
                self.put_food()
                self.set_direction()
                self.move()
            else:
                self.set_direction()
                self.move()
            return True

    def set_direction(self, direction=[0, 0, 0, 0]):
        if direction == [1, 0, 0, 0] and self.direction != [0, 0, 1, 0]:
            self.direction = [1, 0, 0, 0]
        elif direction == [0, 1, 0, 0] and self.direction != [0, 0, 0, 1]:
            self.direction = [0, 1, 0, 0]
        elif direction == [0, 0, 1, 0] and self.direction != [1, 0, 0, 0]:
            self.direction = [0, 0, 1, 0]
        elif direction == [0, 0, 0, 1] and self.direction != [0, 1, 0, 0]:
            self.direction = [0, 0, 0, 1]

    def move(self):

        x1 = self.head.xcor()
        y1 = self.head.ycor()

        for n in self.bodies:
            x2, y2 = n.xcor(), n.ycor()
            n.goto(x1, y1)
            x1 = x2
            y1 = y2

        if self.direction == [1, 0, 0, 0]:
            y = self.head.ycor()
            self.head.sety(y + 20)
        elif self.direction == [0, 1, 0, 0]:
            x = self.head.xcor()
            self.head.setx(x + 20)
        elif self.direction == [0, 0, 1, 0]:
            y = self.head.ycor()
            self.head.sety(y - 20)
        elif self.direction == [0, 0, 0, 1]:
            x = self.head.xcor()
            self.head.setx(x - 20)

        self.wn.update()

    def put_head(self):
        head = turtle.Turtle()
        head.speed(0)
        head.shape("square")
        head.color("black")
        head.penup()
        head.goto(0, 0)
        head.direction = "stop"

        self.food = turtle.Turtle()
        self.food.speed(0)
        self.food.shape("circle")
        self.food.color("red")
        self.food.penup()

        self.head = head
        self.direction = [1, 0, 0, 0]

    def put_food(self):
        self.food.goto(random.randint(-self.game_size/2+20, self.game_size/2-20),
                       random.randint(-self.game_size/2+20, self.game_size/2-20))


play = Snake()

# bg_tasks = tr.Thread(target=agent, args=(play,))
# bg_tasks.daemon = True
# bg_tasks.start()

Agent1 = agent(play)
