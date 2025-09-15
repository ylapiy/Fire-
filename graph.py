class ListaAdjacencia:
    
      def __init__ (self, vertices):
          self.vertices = vertices
          self.grafo = [[] for i in range(self.vertices)]
        
      def Adiciona_Aresta(self, u, v):
          self.grafo[u-1].append(v)
          self.grafo[v-1].append(u)

      def mostra_Lista(self):
          for i in range(self.vertices):
              print(f'{i+1}', end='  ')
              for j in self.grafo[i]:
                print(f'{j} ->',end ='  ')
              print(' ')
          
class MatrizAdjacencia:
    
    def __init__ (self, vertices):
        self.vetices = vertices
        self.grafo = [[0]*self.vetices for i in range(self.vetices)]
    
    def adicionar_aresta(self,u,v,p):
        if not(self.grafo[u-1][v-1] == 1 or self.grafo[v-1][u-1] == 1 ):
            self.grafo[u-1][v-1] = p
            self.grafo[v-1][u-1] = p 
        else:
            print(f'ja conectado! : {u} e {v}')
        
    def mostra_matriz(self):
        print("Matriz :")
        for i in range(self.vetices):
            print(self.grafo[i])

    def Dijkstra():

class DicionarioAdjacencia:
    
    def __init__ (self):    
        self.grafo = {}
        
    def AdicionarPonto(self,u):
        if u is not self.grafo: self.grafo[u] = []
        else : print("ja adicionado!")
    
    def AdicionarAresta(self,u,v,p):
        if v is not self.grafo[u]: self.grafo[u].append([v,p])
        else : print("Já Adicionado!!!")