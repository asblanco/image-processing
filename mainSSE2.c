#include <stdio.h>
#include <stdlib.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <emmintrin.h>

typedef union um128 {
    __m128i mm;
    long long lli[2];
    unsigned long long ulli[2];
    int si[4];
    unsigned int usi[4];
    short hi[8];
    unsigned short uhi[8];
    signed char qi[16];
    unsigned char uqi[16];
} __attribute__((aligned(16))) um128;

//Las secciones deben ser multiplos de 16, ya que el registro sse almacena 16 bytes
int altoSeccion = 32;
int anchoSeccion = 32;

/*img --> es una imagen en 3 canales
 imgB --> es una imagen con tamaño de una sección que contiene la seccion de la imagen del medio que se pretende buscar
 La funcion guarda, mediante el puntero p, en un vector la fila y columna de img donde empieza la seccion
    mas parecida a imgB. Ademas tambien almacena el valor del error calculado*/
void compararSecciones (IplImage *img, IplImage *imgB, int *p);

/*Copia una seccion de la imagen A en la imagen B*/
void copiarSeccion (IplImage *imgA, IplImage *imgB, int filaA, int columnaA, int filaB, int columnaB);

/*Convierte una imagen de 3 canales a 4*/
void convertir4canales (IplImage *img3c, IplImage *img4c);

/*Convierte una imagen de 4 canales a 3 canales*/
void convertir3canales (IplImage *img3c, IplImage *img4c);

int main(int argc, char** argv) {

    if (argc != 4) {
        printf("Usage: %s image_file_name\n", argv[0]);
        return EXIT_FAILURE;
    }

    //Carga las imagenes originales
    IplImage* imgI = cvLoadImage(argv[1], CV_LOAD_IMAGE_UNCHANGED); //Imagen I izquierda
    IplImage* imgD = cvLoadImage(argv[2], CV_LOAD_IMAGE_UNCHANGED); //Imagen D derecha
    IplImage* imgM = cvLoadImage(argv[3], CV_LOAD_IMAGE_UNCHANGED); //Imagen M que se va a comparar y hacer una imagen B
    //Crea imagen del mismo tamaño, será la imagen resultante
    IplImage *imgB = cvCreateImage(cvGetSize(imgM), IPL_DEPTH_8U, 3); //Imagen B

    /*Imagen del tamaño de una zona en la que se va a ir guardando la zona a comparar*/
    IplImage* ImgZonaM = cvCreateImage(cvSize(anchoSeccion,altoSeccion) ,IPL_DEPTH_8U, 3);
    IplImage* ImgZonaM4 = cvCreateImage(cvSize(anchoSeccion,altoSeccion) ,IPL_DEPTH_8U, 4); //Imagen en 4 canales

    //Comprobar que existen los archivos
    if (!imgI) {
        printf("Error: fichero %s no leido\n", argv[1]);
        return EXIT_FAILURE;
    }

    if (!imgD) {
        printf("Error: fichero %s no leido\n", argv[2]);
        return EXIT_FAILURE;
    }

    if (!imgM) {
        printf("Error: fichero %s no leido\n", argv[3]);
        return EXIT_FAILURE;
    }

    //Muestra las tres imagenes iniciales
    cvNamedWindow("Imagen Izquierda", 1);
    cvShowImage("Imagen Izquierda", imgI);
    cvWaitKey(0);

    cvNamedWindow("Imagen Derecha", 1);
    cvShowImage("Imagen Derecha", imgD);
    cvWaitKey(0);

    cvNamedWindow("Imagen Medio", 1);
    cvShowImage("Imagen Medio", imgM);
    cvWaitKey(0);

    //Declarar la variable del fichero
    FILE *file;
    //Crear el fichero llamado imagenB para escribir en el (w)
    file = fopen ("imagenB.txt", "w");

    /*Vector que sera rellenado por la funcion compararSecciones. La primera posicion es la X y la segunda la Y
    X e Y son las posiciones(fila y columna) de la seccion mas parecida en la imagen derecha o izquierda
     La tercera posicion almacena el resultado de la comparacion de la seccion mas parecida (el error cuadratico)*/
    int vectorI[3];
    int vectorD[3];
    //Crea punteros a los vectores. El puntero se le pasa a la funcion compararSecciones para que acceda al vector
    int *pI= vectorI;
    int *pD= vectorD;

    int fila, columna;
    //Recorre por secciones la imagen del medio, la que se quiere conseguir a partir de la izquierda y derecha
    for (fila = 0; fila < imgM->height; fila += altoSeccion) {
        for (columna = 0; columna < imgM->width; columna += anchoSeccion) {
            copiarSeccion(imgM, ImgZonaM, fila, columna, 0, 0); //Copia la seccion actual en la imagen ImgZonaM de 3 canales
            convertir4canales(ImgZonaM, ImgZonaM4); //Convierte ImgZonaM a 4 canales
            //Se pasan las imagenes imgI e imgD en 3 canales ya que la funcion las pasa a 4
            compararSecciones (imgI, ImgZonaM4, pI); //Busca la seccion mas parecia en la imagen izquierda
            compararSecciones (imgD, ImgZonaM4, pD); //Busca la seccion mas parecia en la imagen derecha

            //Compara los errores obtenidos y el menor es el copiado en la imagen final (imgB)
            if(pI[2] < pD[2]){
                copiarSeccion(imgI, imgB, pI[0], pI[1], fila, columna);
                fprintf(file,"Img Izquierda: F-%d C-%d\r\n", pI[0], pI[1]);
            } else {
                copiarSeccion(imgD, imgB, pD[0], pD[1], fila, columna);
                fprintf(file,"Img Derecha: F-%d C-%d", pD[0], pD[1]);

                //Añade un retorno de carro al fichero
                char enter[]="\r\n";
                const char *ent = enter;
                fputs(ent, file);
            }
        }
    }

    //Cierra el fichero .txt
    fclose(file);

    //Muestra la imagen tipo B final
    cvNamedWindow("imagenB", 1);
    cvShowImage("imagenB", imgB);
    cvWaitKey(0);

    // memory release for img before exiting the application
    cvReleaseImage(&imgI);
    cvReleaseImage(&imgD);
    cvReleaseImage(&imgM);
    cvReleaseImage(&imgB);

    cvDestroyWindow("Imagen Izquierda");
    cvDestroyWindow("Imagen Derecha");
    cvDestroyWindow("Imagen Medio");
    cvDestroyWindow("imagenB");

    return EXIT_SUCCESS;
}

//Se le pasa la imagen Izquierda o Derecha en 3 canales, y la imagen con el tamaño de la sección a comparar de la imagen del medio
void compararSecciones (IplImage *img, IplImage *imgB, int *p){
    /*fila y columna: se usan en los bucles
     *minimo: almacena el error cuadratico minimo al ir comparando las secciones
     *dif: almacena el error de una seccion, cada seccion nueva se inicializa a 0
     filMin y colMin: almacenan la fila y columna de la seccion mas parecida*/
    int fila, columna, minimo, filaMin, colMin;
    float dif;
    minimo = 67108864; //2^26
    /*regA y regB almacenan el registro sse intacto
     compA y compB almacenan 3 componentes, cada componente en un int, el cuarto es alfa*/
    um128 regA, regB, compA, compB;

    //Recorre la imagen A para hayar la seccion mas parecida. Recorre por secciones
    for (fila = 0; fila < img->height; fila += altoSeccion) {
        for (columna = 0; columna < img->width; columna += anchoSeccion) {
            /*Crea una imagen del tamaño de una sección de la imagen A para recorrerla*/
            IplImage* ImgZonaA = cvCreateImage(cvSize(anchoSeccion, altoSeccion), IPL_DEPTH_8U, 3);
            IplImage* ImgZonaA4 = cvCreateImage(cvSize(anchoSeccion, altoSeccion), IPL_DEPTH_8U, 4); //La convierte a 4 canales
            copiarSeccion(img, ImgZonaA, fila, columna, 0, 0); //Copia la seccion de la imagen A en imgZonaA de 3 canales, ya que ambas son de 3 canales
            convertir4canales(ImgZonaA, ImgZonaA4); //Pasa ImgZonaA a 4 canales para poder realizar la comparacion

            dif = 0;

            /*acumuladorA y acumuladorB: guardan la suma de los errores*/
            um128 acumuladorA, acumuladorB;
            acumuladorA.mm = _mm_set1_epi32(0);
            acumuladorB.mm = _mm_set1_epi32(0);

            int i,j;
            __m128i *pImgA = (__m128i *) ImgZonaA4->imageData; //Recorre la imagen A
            __m128i *pImgB = (__m128i *) imgB->imageData; //Recorre la imagen del medio

            /*Recorre las dos zonas de A y de B para compararlas hayando el error cuadratico
             Va de 16 en 16 debido a los registros sse*/
            for (i = 0; i < ImgZonaA4->imageSize; i += 16) {
                regA.mm = _mm_load_si128(pImgA); //Almacena 16 bytes de img
                regB.mm = _mm_load_si128(pImgB); //Almacena 16 bytes de imgB
                //Bucle para recorrer el registro sse como 4 ints
                for (j = 0; j < 4; j++) {
                    //Una componente de A en cada int
                    compA.mm = _mm_setr_epi8(regA.uqi[j*4],   0, 0, 0,
                                             regA.uqi[j*4+1], 0, 0, 0,
                                             regA.uqi[j*4+2], 0, 0, 0,
                                             0, 0, 0, 0);

                    //Una componente de B en cada int
                    compB.mm = _mm_setr_epi8(regB.uqi[j*4],   0, 0, 0,
                                             regB.uqi[j*4+1], 0, 0, 0,
                                             regB.uqi[j*4+2], 0, 0, 0,
                                             0, 0, 0, 0); 

                    compA.mm = _mm_sub_epi32(compA.mm, compB.mm); //Se resta cada int, es decir, componenteA - componenteB
                    compA.mm = _mm_mullo_epi16(compA.mm, compA.mm); //Se multiplican shorts ya que da como resultado int. Asi se eleva al cuadrado

                    //Se divide el registro compA en dos, poniendo cada short en un int, ya que al realizar las sumas consecutivas excede el tamaño de un short
                    //y es necesario guardar la suma de cada componente en un int
                    __m128i a = _mm_setr_epi16(compA.uhi[0], 0,
                                               compA.uhi[1], 0,
                                               compA.uhi[2], 0,
                                               compA.uhi[3], 0); //Se guardan los 4 primeros shorts de compA

                    __m128i b = _mm_setr_epi16(compA.uhi[4], 0,
                                               compA.uhi[5], 0,
                                               compA.uhi[6], 0,
                                               compA.uhi[7], 0); //Se guardan los 4 ultimos shorts de compA

                    acumuladorA.mm = _mm_add_epi32(acumuladorA.mm, a); //Se suman sin saturación
                    acumuladorB.mm = _mm_add_epi32(acumuladorB.mm, b);
                }
                //Aumenta los punteros para apuntar al siguiente registro sse a comparar
                pImgA++;
                pImgB++;
            }

            //Se suman todas las componentes de los dos registros que almacenan las sumas de toda la zona
            dif = acumuladorA.usi[0] + acumuladorA.usi[1] + acumuladorA.usi[2] + acumuladorA.usi[3]
                + acumuladorB.usi[0] + acumuladorB.usi[1] + acumuladorB.usi[2] + acumuladorB.usi[3];
            
            //Almacena la seccion mas parecida
            if(dif < minimo){
                minimo = dif;
                colMin = columna;
                filaMin = fila;
            }
        }
    }

    //Guarda en el vector los datos de la seccion mas parecida una vez acabada la comparacion
    p[0] = filaMin;
    p[1] = colMin;
    p[2] = minimo;
}

void copiarSeccion (IplImage *imgA, IplImage *imgB, int filaA, int columnaA, int filaB, int columnaB){
    int fila, columna;
    int diferencia = filaB - filaA;

    for (fila = filaA; fila < (filaA + altoSeccion); fila++) {
        uchar *pImgA = (uchar *) imgA->imageData + fila * imgA->widthStep + columnaA*3;
        uchar *pImgB = (uchar *) imgB->imageData + (fila+diferencia) * imgB->widthStep + columnaB*3;
        for (columna = columnaA; columna < (columnaA + anchoSeccion); columna++) {
            *pImgB++ = *pImgA++;
            *pImgB++ = *pImgA++;
            *pImgB++ = *pImgA++;
        }
    }
}

void convertir4canales(IplImage *img3c, IplImage *img4c) {
    int fila, columna;
    for (fila = 0; fila < img4c->height; fila++) {
        unsigned char *pImg4 = (unsigned char *) img4c->imageData + fila * img4c->widthStep;
        unsigned char *pImg3 = (unsigned char *) img3c->imageData + fila * img3c->widthStep;
        for (columna = 0; columna < img4c->width; columna++) {
            *pImg4++ = *pImg3++;
            *pImg4++ = *pImg3++;
            *pImg4++ = *pImg3++;
            *pImg4++ = 0;
        }
    }
}

void convertir3canales(IplImage *img3c, IplImage *img4c) {
    int fila, columna;
    for (fila = 0; fila < img4c->height; fila++) {
        unsigned char *pImg4 = (unsigned char *) img4c->imageData + fila * img4c->widthStep;
        unsigned char *pImg3 = (unsigned char *) img3c->imageData + fila * img3c->widthStep;
        for (columna = 0; columna < img4c->width; columna++) {
            *pImg3++ = *pImg4++;
            *pImg3++ = *pImg4++;
            *pImg3++ = *pImg4++;
            *pImg4++;
        }
    }
}