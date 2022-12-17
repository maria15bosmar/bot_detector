from main import prob

if __name__ == "__main__":
    l_user = ["_guille_3", "maria15bm", "boliviiic", "maremotoY", "lostinmiriam", "martaguiraoa"]
    num_mod = 1
    label_names = ["humano", "bot"]
    now = 0
    aciertos = 0
    for el in l_user:
        if prob(el, num_mod) == now:
            aciertos+=1


    print(f"Se acertó {aciertos} de {len(l_user)} de la clase {label_names[now]}")
    print(f"Porcentaje {(aciertos/len(l_user))*100}")



