import cv2
import numpy as np
import math

def main():
    #deklaracja zmiennych pomocniczych
    str1= "minuta: "
    str2= "godzina: "
    c1=6; c2=30 # 1 minuta = 6 stopni, 1h= 30 stopni
    j1=15; j2=60; j3=2; j4=11 # j1,j2 minuty do iteracji, j3,j4 godziny do iteracji
    clock_min = 0; clock_h =1;
    # deklaracja zmiennych przechowujacych nazwe i format zdjecia
    name = [str(i) for i in range(1, 31 )]
    format = '.jpg'
    for i in name: #glowna petla programu iterujaca zdjecia
        print(i) #numer zdjecia
        c = i + format #nazwa pliku
        img = cv2.imread(c, 0) #wczytanie zdjecia
        img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1) #zmiana wielksci obrazu
        img_blur = cv2.medianBlur(img, 7)  # rozmycie zdjecia
        cimg = cv2.cvtColor(img_blur, cv2.COLOR_GRAY2BGR) # zdjecie w odcieniach
        # szarosci
        cimg2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        _, thersh1 = cv2.threshold(cimg, 150, 255, cv2.THRESH_BINARY_INV)
        linesP,circles,cdst = kola(cimg2, img)  #wywolanie funkcji wykrywajacej
        if linesP is not None:
            # kolo i wskazowki zegara
            tab_l, tab_a, tab_d = linie(linesP, circles,cdst) #wywolanie funkcji
            # wyliczajacej katy wskazowek
            cdst,x,y,z = sort_function(tab_l,tab_a,tab_d,cdst) #wywolanie funkcji
            # sortujacej linie
            min = minuta(x[0], c1, str1,j1,j2,clock_min) # wywolanie funkcji liczacej
            # minuty
            h = minuta(x[1], c2, str2,j3,j4,clock_h) # wywolanie funkcji liczacej
            # godzine
            czas= str(h) +":" + str(min) # string przechowujacy godzine ze zdjecia
            print (czas) # wyswietlenie w teminalu godziny

            font = cv2.FONT_HERSHEY_SIMPLEX #font wyswietlanego tekstu na zdjeciu
            org = (50, 50)  # wielkosc fontu
            fontScale = 1  # skala fontu
            color = (255, 0, 0) # kolor niebieski w BGR
            thickness = 2 # grubosc linii rowna 2 px
            # Uzycie metody puttext w celu wyswietlenia tekstu na zdjeciu
            image = cv2.putText(img, czas, org, font,
                                fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow('detected circles', image) #wyswietlenie zdjecia
            cv2.waitKey(0) # czekanie na przycisk
            cv2.destroyAllWindows() # wylaczenie zdjecia
        else:
            print("Pominieto")

def kola(cimg2,img):
    edges = cv2.Canny(cimg2, 100, 200) # wykrycie krawedzi
    # cv2.imshow('canny_filter', edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # wykrycie okregow na zdjeciu
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 300,
                               param1=50, param2=30, minRadius=30, maxRadius=200)
    print(circles)
    if circles is not None:
        circles = np.uint16(np.around(circles)) # zmiana typu zmiennych tablicy okregu
        diameter = [] # deklaracja tablicy do przechowywania srednicy okregu
        for k in circles[0, :]:
            diameter.append(k[2]) # wypelnienie tablicy srednica
            if max(diameter) == k[2]: # wykrycie najwiekszego okregu
                circles = k # ograniczenie sie do jednego okregu
        # wyrysowanie okregu na zdjeciu
        cv2.circle(cimg2, (circles[0], circles[1]), circles[2], (0, 255, 0), 2)
        # zaznaczenie na zdjeciu srodka okregu
        cv2.circle(cimg2, (circles[0], circles[1]), 2, (0, 0, 255), 3)

        mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8) # czarne tlo
        # wyciecie w  tle okregu
        cv2.circle(mask, (circles[0], circles[1]), circles[2], (255, 255, 255), -1)
        # nalozenie 2 zdjec
        fg = cv2.bitwise_or(img, img, mask=mask)
        # cv2.imshow('fg', fg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        edges = cv2.Canny(fg, 100, 200) # wykrycie krawedzi

        # wykrycie linii w okregu
        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, None, 30, 10)
        cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) # zdjecie w odcieniach
        # szarosci
        if linesP is not None:
            return (linesP,circles,cdst) # zwrocenie parametrow linii, okregu i zdjecia
        else:
            print("None lines in circle")  # komunikat o braku okregow na zdjeciu
            linesP = None
            circles = None
            cdst = None
            return (linesP, circles, cdst)
    else:
        print("None objects in circles") # komunikat o braku okregow na zdjeciu
        linesP = None
        circles = None
        cdst=None
        return(linesP,circles,cdst)

def linie(linesP,circles,cdst):
    tab_d =[]
    tab_l =[]
    tab_a =[]
    if linesP is not None:
        for j in range(0, len(linesP)):
            l = linesP[j][0] #parametry linii
            x = np.sqrt((circles[0] - l[0]) ** 2 + (circles[1] - l[1]) ** 2) # odlegosc jednego konca wskazowki od srodka okregu
            y = np.sqrt((circles[0] - l[2]) ** 2 + (circles[1] - l[3]) ** 2) # odlegosc drugiego konca wskazowki od srodka okregu
            d = np.sqrt((l[0] - l[2]) ** 2 + (l[1] - l[3]) ** 2) #dlugosc odcinka
            if (x <= 40) or (y <= 40): #pominiecie linii od srodka okregu o odleglosci wiekszej niz podana
                tab_d.append(d) #dodanie parametru dlugosci wskazowki do tablicy
                if x < y:
                    # obliczenie kata nachylenia wskazowki zegara w zaleznosci od tego
                    # ktora strona wskazowki jest blizej srodka okregu
                    angle = int(math.atan2(-(l[3] - circles[1]), l[2] - circles[0]) * 180 / math.pi)
                else:
                    angle = int(math.atan2(-(l[1] - circles[1]), l[0] - circles[0]) * 180 / math.pi)
                if angle < 0: # gdy kat ujemny dodaj 360 stopni
                    angle += 360
                tab_l.append(l) # dodaj parametry wskazowki do tablicy
                tab_a.append(angle) # dodaj kat do tablicy
            else:
                pass # jesli wskazowka oddalona od srodka okregu, pomin
    else:
        print("no lines detected") # gdy nie wykryto linii wypisz komunikat
    return (tab_l,tab_a,tab_d) # zwroc utworzone tablice

def sort_function(tab_l,tab_a,tab_d,cdst): #fukcja filtrujaca linie po dlugosci i kacie
    x=[]
    tab_L=[] #tablice do przechowywania kolejno,parametrow wskazowki, kata i dlugosci wskazowki
    tab_A=[] 
    tab_D=[]
    y = []
    z= []
    list={};list2={} #slowniki by moc sortowac po dlugosci linie i nie stracic jej pozostalych parametrow
    for i in range(len(tab_a)):
        list[round(tab_d[i],2)]= tab_a[i] #uzaleznienie dl. wskazowki od jej kata
        list2[round(tab_d[i],2)]= tab_l[i] #uzaleznienie dl. wskazowki od jej wektora parametrow
    for i in sorted(list, reverse = True):
        tab_D.append(i) #posortowane dlugosci linii
        tab_A.append(list[i]) # dodanie do tablicy katow posortowanych linii
    for i in sorted(list2, reverse = True):
        tab_L.append(list2[i]) # analogicznie dodane do tablicy wektor parametrow posortowanych linii
        tab_a = tab_A
        tab_d = tab_D
        tab_l = tab_L

    #pominiecie linii o zblizonym kacie
    iter = 0
    for i in tab_a:
        tab = []
        if len(x) == 0: # jesli tablica jest pusta uzupelnij pierwsza wartosciami [kat, wektor linii,dlugosc]
            x.append(tab_a[0]); y.append(tab_l[iter]),z.append(tab_d[iter])
        else:
            for j in range(len(x)):
                # zbadaj roznice katow gdy mniejsze niz 20 stopni lub wieksze jak 350,  dodaj do tab jedynke
                if (abs(i-x[j])< 20) or (abs(i-x[j])>350) :
                    tab.append(1)
                # inaczej dodaj do tab 0
                else:
                    tab.append(0)
            # gdy 1 reprezentujaca zblizony kat, pomin linie
            if 1 in tab:
                pass
            # gdy 0 i w tablicy x mniej niz jedna wskazowka dodaj parametry linii do tablic
            else: #
                if len(x) <= 2:
                    x.append(i); y.append(tab_l[iter]) #tab x- kat, tab y wektor linii, tab z -dl. linii
                    z.append(tab_d[iter])
        iter += 1

    for i in range(len(x)):
        print(x[i],y[i]) # wyswietl parametry linii
        l = y[i]
        #narysuj na zdjeciu linie, nastepnie wyswietl zdjecie
        cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('sorted  ', cdst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if len(x) > 2:  # gdy wiecej niz 2 wskazowki, pozbadz sie parametrow tej najdluzszej (sek)
        del x[0]
        del y[0]
        del z[0]
    return cdst,x,y,z #zwroc parametry wskazowek zegara i zdjecie zegara

def minuta(angle, c, str,t1,t2,x): #odczyt wskazowek zegara
    if angle >= 0 and angle <=90: # gdy kat pomiedzy podanymi przypisz parametry do iteracji
        i =0; i_old=0; j=t1 # i obecnie badany kat, i_old poprzedni kat, j- czas (min lub h) przy kacie = 0
        while i < 90: # gdy kat mniejszy od 90
            i+= c # inkrementuj co krok c (6-minuty lub 30-godziny) stopni
            if angle >= i_old and angle < i: # jesli kat mniejszy miesci sie miedzy iterowanym przedzialem
                minuta = j # czas rowny j
                print(str,minuta) #wyswietl czas w terminalu
                if (x <= 0)  & (minuta >= 60): # jesli liczone sa minuty (x = 0) i minuty wieksze jak 60
                    minuta = 0; # minuty = 0
                elif (x>=0) & (minuta<=0): # gdy wywolana funkcja godziny i zmienna minuty mniejsza od zera
                    minuta = 12; #godzina jest rowna 12
                break
            j -= 1 # dekrementacja czasu
            i_old = i #przypisanie ostatniego kata
    else:
        i =90; i_old=90; j=t2; # gdy kat wiekszy niz 90  zacznij iteracje kata od i = 90, pozostale analogicznie
        while i < 360: # gdy kat mniejszy niz 360
            i+= c # inkrementuj kat co krok c
            if angle >= i_old and angle < i: # analogicznie odczytaj czas
                minuta = j
                if minuta >= 60: # gdy minuty = 60, przypisz 0
                    minuta = 0;
                print(str,minuta)
                break
            j -= 1
            i_old = i
    return minuta # zwroc parametr przechowujacy czas zegara

if __name__ == "__main__":
    main()