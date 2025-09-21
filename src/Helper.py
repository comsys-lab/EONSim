import time

class Helper:
    def init(self):
        self.start = 0
        self.end = 0
        
    def set_timer(self):
        self.start = time.perf_counter()
    
    def end_timer(self, task):
        self.end = time.perf_counter()
        print('(Time elapsed(s) in {}: {:10.6f}sec)'.format(task, self.end-self.start))

def print_styled_header(title):
    width = 100
    title_decorated = f"《 {title} 》"
    print("\n╔" + "═"*width + "╗")
    print("║" + "▒"*(width//2-len(title_decorated)//2-1) + title_decorated + "▒"*(width//2-len(title_decorated)//2-1) + "║")
    print("╚" + "═"*width + "╝")

def print_styled_box(title, content_lines):
    width = 100
    title_decorated = f"《 {title} 》"
    print("\n╔" + "═"*width + "╗")
    print("║" + "▒"*(width//2-len(title_decorated)//2-1) + title_decorated + "▒"*(width//2-len(title_decorated)//2-1) + "║")
    print("╠" + "═"*width + "╣")
    for line in content_lines:
        print("║ " + line.ljust(width-2) + " ║")
    print("╚" + "═"*width + "╝")