from annotatePDF import *
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from PyInstaller.utils.hooks import collect_data_files

def functiona():
    print("a")

def loadPDFfile(master):
    master.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                                 filetypes=(("PDF files", "*.pdf"), ("all files", "*.*")))
    path = master.filename
    weight_v = "configs/Glycan_300img_5000iterations.weights"
    print("Loaded file:", path, type(path))
    try:
        checkpath()
    except PermissionError:
        time.sleep(2)
        checkpath()

    annotatePDFGlycan(path, weight_v)

    print("Scrip Finished annotated:",path)
    os.startfile(r"test\\0000page.pdf")

def loadImagefile(master):
    # Instruct pyinstaller to collect data files from resources package.
    datas = collect_data_files('pygly3')
    master.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                                 filetypes=(("png files", "*.png"),("jpeg files", "*.jpg"), ("all files", "*.*")))

    path =master.filename
    img_file = cv2.imread(path)
    monoCount_dict, final, origin, mask_dict, return_contours = countcolors(img_file)
    mono_dict, a, b = extractGlycanTopology(mask_dict, return_contours, origin)
    for mono_id in mono_dict.keys():
        print(mono_id, mono_dict[mono_id][4])
    print(a.shape)
    cv2.imshow('a', cv2.resize(a, None, fx=1, fy=1))
    cv2.waitKey(0)
    cv2.imshow('b', cv2.resize(b, None, fx=2, fy=2))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    glycoCT=buildglycan(mono_dict)
    print("Condensed GlycoCT:\n", glycoCT)
    #select place to save
    folder_selected = filedialog.askdirectory()
    print(folder_selected)
    cv2.imwrite(f"{folder_selected}/annotated_a.png",a)
    cv2.imwrite(f"{folder_selected}/annotated_b.png",b)
    f = open(f"{folder_selected}/GlycoCT.txt", "w+")
    f.write(glycoCT)
    f.close()
    os.startfile(f"{folder_selected}/GlycoCT.txt")

    accession = searchGlycoCT(glycoCT)
    f = open(f"{folder_selected}/Accession_hits.txt", "w+")

    f.write(f"{accession}\nhttps://gnome.glyomics.org/StructureBrowser.html?focus={accession}")
    f.close()
    os.startfile(f"{folder_selected}/Accession_hits.txt")

if __name__ == '__main__':
    #findImageonDesktopVisual()
    f = open("run.bat", "w+")
    f.write("@ECHO OFF\nECHO Nhat Duong. Glycan extractor \nmain.exe\nPAUSE")
    f.close()
    #GUIp
    master = Tk()
    master.title("Glycan extractor")

    BKbotIcon=PhotoImage(file="Data/Images/BKbotIcon_small.png")
    background = PhotoImage(file="Data/Images/BKbotBackground_dim.png")
    background_label = Label(master, image=background)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    master.iconphoto(False, BKbotIcon)
    # sets the geometry of
    # main root window
    master.geometry("672x480")
    label = Label(master, text="Welcome to Nhat Duong's Glycan Extraction Tool")

    label.pack(side=TOP, pady=10)
    # a button widget which will
    # open a new window on button click
    #cairos_thread = threading.Thread(target=runloop)
    #riftb_thread = threading.Thread(target=farmrift)
    # make test_loop terminate when the user exits the window
    #cairos_thread.daemon = True
    #riftb_thread.daemon = True

    #c_btn = Button(master,text="Click to run Cairos.",command=lambda: cairos_thread_func(cairos_thread))

    #r_btn = Button(master,text="Click to run Rift Beast.", command=lambda: riftb_thread_func(riftb_thread))

    # Following line will bind click event
    # On any click left / right button
    # of mouse a new window will be opened



    pdf_btn = Button(master, text="Open pdf file",command=lambda: loadPDFfile(master))
    image_btn = Button(master, text="Open image file", command=lambda: loadImagefile(master))
    quit_btn = Button(master, text="Quit",  width=8,
                         command=master.quit)

    pdf_btn.pack(pady=10)
    image_btn.pack(pady=10)
    quit_btn.pack(pady=10)


    # mainloop, runs infinitely
    mainloop()
