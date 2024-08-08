library(tidyverse)
library(palmerpenguins)
library(ggthemes)

x <- rnorm(100)
y <- rnorm(100)
data <- tibble(x = x, y = y)

plot(x, y, xlab = "this is the x-axis", ylab = "this is the y-axis", main = "Plot of X vs Y")

pdf("Figure.pdf") # can also use jpeg() here
plot(x, y, col = "green")
dev.off()

ggplot(data, aes(x = x, y = y)) +
    geom_point() +
    labs(
        title = "Plot of X vs Y",
        x = "this is the x-axis",
        y = "this is the y-axis"
    )

ggsave("Figure.pdf")

ggplot(
    data = penguins,
    mapping = aes(x = flipper_length_mm, y = body_mass_g)
) +
    geom_point(mapping = aes(color = species, shape = species)) +
    geom_smooth(method = "lm") +
    labs(
        title = "Body mass and flipper length",
        subtitle = "Dimensions for Adelie, Chinstrap, and Gentoo Penguins",
        color = "Species", shape = "Species"
    ) +
    scale_color_colorblind()

x <- seq(-pi, pi, length = 50)
y <- x
f <- outer(x, y, function(x, y) cos(y) / (1 + x^2))
contour(x, y, f)

data <- tibble(expand.grid(x, y)) # basically equivalent of outer()
data <- data %>%
    transmute(
        x = Var1,
        y = Var2,
        z = cos(y) / (1 + x^2))
ggplot(data, aes(x, y, z = z)) + geom_contour()

contour(x, y, f, nlevels = 45)
fa <- (f - t(f)) / 2
contour(x, y, fa, nlevels = 15)
image(x, y, fa)
persp(x, y, fa)
persp(x, y, fa, theta = 30)
persp(x, y, fa, theta = 30, phi = 20)
persp(x, y, fa, theta = 30, phi = 70)
persp(x, y, fa, theta = 30, phi = 40)

data <- data %>%
    transmute(
        x,
        y,
        z,
        zt = cos(x) / (1 + y^2),
        za = (z - zt) / 2
    )
ggplot(data, aes(x, y, z = za)) + geom_contour()
ggplot(data, aes(x, y, z = za)) + geom_contour_filled()
# To do the 3d plots using ggplot, we'd have to do some work and I don't feel like it

Auto <- read.table("StatisticalLearning\\Auto.data", header = T, na.strings = "?", stringsAsFactors = T)
View(Auto)
head(Auto)
# Auto.data is a weird TSV file with spaces for the actual rows, so eh, I'll skip loading it using tidyverse

Auto <- read.csv("StatisticalLearning\\Auto.csv", na.strings = "?", stringsAsFactors = T)
View(Auto)
dim(Auto)
Auto[1:4,]
Auto <- na.omit(Auto)
dim(Auto)
Auto$cylinders <- as.factor(Auto$cylinders)
names(Auto)
# This only works when attached
# attach(Auto)
# plot(cylinders, mpg)
# detach()
hist(Auto$mpg)
pairs(Auto)
pairs(
    ~ mpg + displacement + horsepower + weight + acceleration,
    data = Auto
)
identify(Auto$horsepower, Auto$mpg, Auto$name)
summary(Auto)
summary(Auto$mpg)

Auto <- read_csv("StatisticalLearning\\Auto.csv", na = c("?")) %>%
    mutate(name = parse_factor(name))
View(Auto)
dim(Auto)
Auto %>% slice_head(n = 4)
Auto <- Auto %>%
    drop_na()
dim(Auto)
Auto <- Auto %>%
    mutate(cylinders = as_factor(cylinders))
names(Auto)
ggplot(Auto, aes(cylinders, mpg)) + geom_point()
ggplot(Auto, aes(cylinders, mpg)) + geom_boxplot()
ggplot(Auto, aes(x = mpg)) + geom_histogram(binwidth = 5) # must specify binwidth
# There is an extra library you can use to get a more ggplot friendly scatterplot matrix, but pairs() works just fine.
pairs(Auto)
pairs(
    ~ mpg + displacement + horsepower + weight + acceleration,
    data = Auto
)
identify(Auto$horsepower, Auto$mpg, Auto$name)
summary(Auto)
summary(Auto$mpg)
