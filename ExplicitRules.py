import random
def spatial(entry, category_a=1, category_b=2):
    [_, spatial, _] = entry
    epsilon = random.randint(-25, 25)
    criterion = 45
    discriminant = spatial - criterion
    max_discriminant = 95 - criterion
    return (category_a if discriminant > epsilon else category_b), abs(discriminant / max_discriminant)


def spatial_and_orientation(entry, category_a=1, category_b=2):
    [_, spatial, ori] = entry
    epsilon = random.randint(-25, 25)
    criterion = 45
    discriminant1 = spatial - criterion
    discriminant2 = ori - criterion
    max_discriminant = 95 - criterion
    return (category_a if (discriminant1 > epsilon) and (discriminant2 > epsilon) else category_b), abs(discriminant1 / max_discriminant)


def orientation(entry, category_a=1, category_b=2):
    [_, _, ori] = entry
    epsilon = random.randint(-25, 25)
    criterion = 45
    discriminant = ori - criterion 
    max_discriminant = 95 - criterion
    return (category_a if discriminant > epsilon else category_b), abs(discriminant / max_discriminant)
