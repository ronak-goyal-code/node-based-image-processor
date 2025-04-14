

#include <QApplication>
#include <QMainWindow>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsRectItem>
#include <QGraphicsLineItem>
#include <QDockWidget>
#include <QLabel>
#include <QFormLayout>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QToolBar>
#include <QMenuBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QVBoxLayout>
#include <QMouseEvent>
#include <QContextMenuEvent>
#include <QGraphicsSceneContextMenuEvent>
#include <QMenu>
#include <QAction>
#include <QPainter>
#include <QtConcurrent>
#include <QFuture>
#include <QFutureWatcher>
#include <QPointF>
#include <QPixmap>
#include <QImage>
#include <QPen>
#include <QBrush>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>
#include <iostream>
#include <cmath>

class MainWindow;  
MainWindow *g_mainWindow = nullptr;


class Node {
public:
    std::string name;
    int id;
    virtual cv::Mat process(const std::vector<cv::Mat>& inputs) = 0;
    virtual bool validate() { return true; }
    virtual ~Node() {}
};

using NodePtr = std::shared_ptr<Node>;


struct Connection {
    int fromNodeIndex;
    int toNodeIndex;
};

class NodeGraph {
public:
    std::vector<NodePtr> nodes;
    std::vector<Connection> connections;
    std::unordered_map<int, cv::Mat> cache;

    int addNode(NodePtr node) {
        int index = static_cast<int>(nodes.size());
        node->id = index;
        nodes.push_back(node);
        return index;
    }

    void addConnection(int fromIndex, int toIndex) {
        connections.push_back({ fromIndex, toIndex });
    }

    bool topologicalSort(std::vector<int>& sortedOrder) {
        std::vector<int> indegree(nodes.size(), 0);
        for (const auto& conn : connections) {
            if (conn.toNodeIndex < static_cast<int>(indegree.size()))
                indegree[conn.toNodeIndex]++;
        }
        std::vector<int> q;
        for (size_t i = 0; i < indegree.size(); ++i)
            if (indegree[i] == 0)
                q.push_back(static_cast<int>(i));
        while (!q.empty()) {
            int index = q.back();
            q.pop_back();
            sortedOrder.push_back(index);
            for (const auto& conn : connections) {
                if (conn.fromNodeIndex == index) {
                    indegree[conn.toNodeIndex]--;
                    if (indegree[conn.toNodeIndex] == 0)
                        q.push_back(conn.toNodeIndex);
                }
            }
        }
        return sortedOrder.size() == nodes.size();
    }

    bool hasCycle() {
        std::vector<int> sorted;
        return !topologicalSort(sorted);
    }



    
    bool processGraph() {
        cache.clear();
        std::vector<int> sorted;
        if (!topologicalSort(sorted)) {
            std::cerr << "Cycle detected! Cannot process the node graph.\n";
            return false;
        }
        // Process nodes sequentially.
        for (int nodeIndex : sorted) {
            std::vector<cv::Mat> inputs;
            for (const auto& conn : connections)
                if (conn.toNodeIndex == nodeIndex)
                    inputs.push_back(cache[conn.fromNodeIndex]);
            cv::Mat result = nodes[nodeIndex]->process(inputs);
            cache[nodeIndex] = result;
        }
        return true;
    }
};

// -----------------------------------------------------------------------------
// Example Node Implementations
// -----------------------------------------------------------------------------

// 1. Image Input Node
class ImageInputNode : public Node {
public:
    std::string filePath;
    cv::Mat image;
    ImageInputNode() { name = "Image Input Node"; }
    bool loadImage(const std::string& path) {
        filePath = path;
        image = cv::imread(path, cv::IMREAD_UNCHANGED);
        return !image.empty();
    }
    cv::Mat process(const std::vector<cv::Mat>&) override {
        return image;
    }
    std::string getMetadata() {
        std::ostringstream meta;
        if (image.empty()) {
            meta << "No image loaded.";
        } else {
            meta << "Dimensions: " << image.cols << " x " << image.rows;
            meta << "\nChannels: " << image.channels();
            meta << "\nType: " << image.type();
        }
        return meta.str();
    }
};

// 2. Output Node
class OutputNode : public Node {
public:
    std::string outputPath;
    std::string format; // "png" or "jpg"
    int quality;
    OutputNode() : quality(95), format("png") { name = "Output Node"; }
    cv::Mat process(const std::vector<cv::Mat>& inputs) override {
        if (inputs.empty() || inputs[0].empty())
            return cv::Mat();
        if (!outputPath.empty()) {
            saveToDisk(inputs[0]);
        }
        return inputs[0];
    }
    bool saveToDisk(const cv::Mat &img) {
        std::vector<int> params;
        if (format == "jpg" || format == "jpeg") {
            params.push_back(cv::IMWRITE_JPEG_QUALITY);
            params.push_back(quality);
        } else if (format == "png") {
            params.push_back(cv::IMWRITE_PNG_COMPRESSION);
            params.push_back(3);
        }
        return cv::imwrite(outputPath, img, params);
    }
};

// 3. Brightness/Contrast Node
class BrightnessContrastNode : public Node {
public:
    int brightness;    // -100 to 100
    double contrast;   // 0.0 to 3.0
    BrightnessContrastNode() : brightness(0), contrast(1.0) { name = "Brightness/Contrast Node"; }
    cv::Mat process(const std::vector<cv::Mat>& inputs) override {
        if (inputs.empty() || inputs[0].empty())
            return cv::Mat();
        cv::Mat output;
        inputs[0].convertTo(output, -1, contrast, brightness);
        return output;
    }
};

// -----------------------------------------------------------------------------
// GUI Classes
// -----------------------------------------------------------------------------

// Forward declaration for NodeGraphicsItem needed for the connection item.
class NodeGraphicsItem;

// ConnectionGraphicsItem: draws a line between two NodeGraphicsItems.
class ConnectionGraphicsItem : public QGraphicsLineItem {
public:
    NodeGraphicsItem *source;
    NodeGraphicsItem *target;
    ConnectionGraphicsItem(NodeGraphicsItem *src, NodeGraphicsItem *tgt)
        : source(src), target(tgt) {
        setPen(QPen(Qt::blue, 2));
        updatePosition();
    }
    void updatePosition();
};

// NodeGraphicsItem: visual representation of a node; supports dragging and context menu.
class NodeGraphicsItem : public QGraphicsRectItem {
public:
    std::shared_ptr<Node> node;
    std::vector<ConnectionGraphicsItem*> connectionsOut;
    
    NodeGraphicsItem(std::shared_ptr<Node> n, QGraphicsItem *parent = nullptr)
        : QGraphicsRectItem(parent), node(n) {
        setRect(0, 0, 160, 80);
        setFlags(QGraphicsItem::ItemIsMovable | QGraphicsItem::ItemIsSelectable);
        setToolTip(QString::fromStdString(n->name));
        setPen(QPen(Qt::black, 2));
        setBrush(QBrush(Qt::lightGray));
    }
    
    // Declaration of overridden context menu event.
    void contextMenuEvent(QGraphicsSceneContextMenuEvent *event) override;
};

// Implementation of ConnectionGraphicsItem::updatePosition()
void ConnectionGraphicsItem::updatePosition() {
    if (source && target) {
        QPointF p1 = source->sceneBoundingRect().center();
        QPointF p2 = target->sceneBoundingRect().center();
        setLine(QLineF(p1, p2));
    }
}

// -----------------------------------------------------------------------------
// NodePropertiesWidget: shows/editable properties for BrightnessContrastNode
// -----------------------------------------------------------------------------
class NodePropertiesWidget : public QWidget {
    Q_OBJECT
public:
    explicit NodePropertiesWidget(QWidget *parent = nullptr) : QWidget(parent) {
        QFormLayout *layout = new QFormLayout(this);
        brightnessSpin = new QSpinBox(this);
        brightnessSpin->setRange(-100, 100);
        layout->addRow("Brightness:", brightnessSpin);
        contrastSpin = new QDoubleSpinBox(this);
        contrastSpin->setRange(0.0, 3.0);
        contrastSpin->setSingleStep(0.1);
        layout->addRow("Contrast:", contrastSpin);
        connect(brightnessSpin, SIGNAL(valueChanged(int)), this, SLOT(brightnessChanged(int)));
        connect(contrastSpin, SIGNAL(valueChanged(double)), this, SLOT(contrastChanged(double)));
    }
    void setBrightnessContrastNode(std::shared_ptr<BrightnessContrastNode> node) {
        bcNode = node;
        if (bcNode) {
            brightnessSpin->setValue(bcNode->brightness);
            contrastSpin->setValue(bcNode->contrast);
        }
    }
signals:
    void propertiesChanged();
private slots:
    void brightnessChanged(int value) {
        if (bcNode) {
            bcNode->brightness = value;
            emit propertiesChanged();
        }
    }
    void contrastChanged(double value) {
        if (bcNode) {
            bcNode->contrast = value;
            emit propertiesChanged();
        }
    }
private:
    std::shared_ptr<BrightnessContrastNode> bcNode;
    QSpinBox *brightnessSpin;
    QDoubleSpinBox *contrastSpin;
};

// -----------------------------------------------------------------------------
// MainWindow: integrates the scene (canvas), dock widgets, and menu actions
// -----------------------------------------------------------------------------
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow(QWidget *parent = nullptr) : QMainWindow(parent) {
        g_mainWindow = this;  // Set global pointer.
        scene = new QGraphicsScene(this);
        view = new QGraphicsView(scene, this);
        setCentralWidget(view);

        pendingConnection = nullptr;
        createMenus();
        createToolbars();
        createPropertiesPanel();
        createPreviewPanel();

        // Create sample nodes.
        auto imgInput = std::make_shared<ImageInputNode>();
        int idx0 = graph.addNode(imgInput);
        addNode(imgInput, QPointF(20, 20));

        auto bcNode = std::make_shared<BrightnessContrastNode>();
        int idx1 = graph.addNode(bcNode);
        addNode(bcNode, QPointF(220, 20));

        auto outNode = std::make_shared<OutputNode>();
        int idx2 = graph.addNode(outNode);
        addNode(outNode, QPointF(420, 20));

        // Connect: Image Input -> Brightness/Contrast -> Output.
        graph.addConnection(idx0, idx1);
        graph.addConnection(idx1, idx2);

        updateGraphAndPreview();
    }
    ~MainWindow() { g_mainWindow = nullptr; }
    
    // Called by NodeGraphicsItem's context menu.
    void handleNodeContextMenu(NodeGraphicsItem *item, const QPoint &globalPos) {
        QMenu menu;
        QAction *deleteAction = menu.addAction("Delete Node");
        QAction *connectAction = menu.addAction("Connect Node");
        QAction *selectedAction = menu.exec(globalPos);
        if (selectedAction == deleteAction) {
            deleteNode(item);
        } else if (selectedAction == connectAction) {
            // If no pending connection, set this item as source.
            if (!pendingConnection) {
                pendingConnection = item;
                QMessageBox::information(this, "Connection", "Select another node to connect to.");
            } else if (pendingConnection != item) {
                // Create connection between pendingConnection and this item.
                addConnection(pendingConnection, item);
                pendingConnection = nullptr;
            } else {
                pendingConnection = nullptr;
            }
        }
    }
    
    // Delete a node and its associated connections.
    void deleteNode(NodeGraphicsItem *item) {
        QList<QGraphicsItem*> toDelete;
        for (QGraphicsItem *gi : scene->items()) {
            ConnectionGraphicsItem *ci = dynamic_cast<ConnectionGraphicsItem*>(gi);
            if (ci && (ci->source == item || ci->target == item))
                toDelete.append(gi);
        }
        for (QGraphicsItem *gi : toDelete)
            scene->removeItem(gi);
        scene->removeItem(item);
        updateGraphAndPreview();
    }
    
    // Add a connection between two node items.
    void addConnection(NodeGraphicsItem *source, NodeGraphicsItem *target) {
        auto connItem = new ConnectionGraphicsItem(source, target);
        scene->addItem(connItem);
        source->connectionsOut.push_back(connItem);
        graph.addConnection(source->node->id, target->node->id);
        updateGraphAndPreview();
    }
    
public slots:
    void openImage() {
        QString fileName = QFileDialog::getOpenFileName(this, "Open Image", "",
                                                        "Images (*.png *.jpg *.bmp)");
        if (!fileName.isEmpty()) {
            bool loaded = false;
            for (auto &node : graph.nodes) {
                if (node->name == "Image Input Node") {
                    auto imgInput = std::dynamic_pointer_cast<ImageInputNode>(node);
                    if (imgInput) {
                        loaded = imgInput->loadImage(fileName.toStdString());
                        if (loaded) {
                            QMessageBox::information(this, "Image Loaded",
                                                     QString::fromStdString(imgInput->getMetadata()));
                            updateGraphAndPreview();
                        } else {
                            QMessageBox::warning(this, "Error", "Could not load image.");
                        }
                    }
                    break;
                }
            }
            if (!loaded)
                QMessageBox::warning(this, "Error", "No Image Input Node found.");
        }
    }
    
    void saveImage() {
        QString fileName = QFileDialog::getSaveFileName(this, "Save Image", "",
                                                        "PNG Image (*.png);;JPEG Image (*.jpg)");
        if (!fileName.isEmpty()) {
            bool saved = false;
            for (auto &node : graph.nodes) {
                if (node->name == "Output Node") {
                    auto outNode = std::dynamic_pointer_cast<OutputNode>(node);
                    if (outNode) {
                        outNode->outputPath = fileName.toStdString();
                        if (graph.processGraph())
                            saved = true;
                    }
                    break;
                }
            }
            if (saved)
                QMessageBox::information(this, "Saved", "Image saved successfully.");
            else
                QMessageBox::warning(this, "Error", "Could not save image.");
        }
    }
    
    void updateGraphAndPreview() {
        if (!graph.processGraph()) {
            QMessageBox::critical(this, "Error", "Error processing the node graph (possible cycle detected).");
            return;
        }
        if (!graph.cache.empty()) {
            int lastIdx = static_cast<int>(graph.cache.size()) - 1;
            cv::Mat result = graph.cache[lastIdx];
            if (!result.empty()) {
                cv::Mat rgb;
                if (result.channels() == 1)
                    cv::cvtColor(result, rgb, cv::COLOR_GRAY2RGB);
                else
                    cv::cvtColor(result, rgb, cv::COLOR_BGR2RGB);
                QImage qimg(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
                previewLabel->setPixmap(QPixmap::fromImage(qimg)
                                        .scaled(previewLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
            }
        }
    }
    
    void addBrightnessNode() {
        auto node = std::make_shared<BrightnessContrastNode>();
        int idx = graph.addNode(node);
        addNode(node, view->mapToScene(view->viewport()->rect().center()));
        updateGraphAndPreview();
    }
    
private:
    QGraphicsScene *scene;
    QGraphicsView *view;
    NodeGraph graph;
    QDockWidget *propertiesDock;
    NodePropertiesWidget *propertiesWidget;
    QDockWidget *previewDock;
    QLabel *previewLabel;
    NodeGraphicsItem *pendingConnection;  // For connection creation

    void createMenus() {
        QMenu *fileMenu = menuBar()->addMenu("&File");
        QAction *openAct = new QAction("&Open Image", this);
        QAction *saveAct = new QAction("&Save Result", this);
        connect(openAct, &QAction::triggered, this, &MainWindow::openImage);
        connect(saveAct, &QAction::triggered, this, &MainWindow::saveImage);
        fileMenu->addAction(openAct);
        fileMenu->addAction(saveAct);
        
        QMenu *nodeMenu = menuBar()->addMenu("&Nodes");
        QAction *addBCAct = new QAction("Add Brightness/Contrast Node", this);
        connect(addBCAct, &QAction::triggered, this, &MainWindow::addBrightnessNode);
        nodeMenu->addAction(addBCAct);
    }
    
    void createToolbars() {
        QToolBar *toolbar = addToolBar("Main Toolbar");
        toolbar->addAction("Open", this, &MainWindow::openImage);
        toolbar->addAction("Save", this, &MainWindow::saveImage);
    }
    
    void createPropertiesPanel() {
        propertiesDock = new QDockWidget("Node Properties", this);
        propertiesWidget = new NodePropertiesWidget(this);
        propertiesDock->setWidget(propertiesWidget);
        addDockWidget(Qt::RightDockWidgetArea, propertiesDock);
        connect(propertiesWidget, SIGNAL(propertiesChanged()), this, SLOT(updateGraphAndPreview()));
    }
    
    void createPreviewPanel() {
        previewDock = new QDockWidget("Preview", this);
        previewLabel = new QLabel(previewDock);
        previewLabel->setAlignment(Qt::AlignCenter);
        previewDock->setWidget(previewLabel);
        addDockWidget(Qt::BottomDockWidgetArea, previewDock);
    }
    
    void addNode(std::shared_ptr<Node> node, const QPointF &pos) {
        NodeGraphicsItem *item = new NodeGraphicsItem(node);
        item->setPos(pos);
        scene->addItem(item);
    }
};

//
// IMPORTANT: Move the implementation of NodeGraphicsItem::contextMenuEvent here,
// after the complete definition of MainWindow to resolve the "undefined type" error.
//
void NodeGraphicsItem::contextMenuEvent(QGraphicsSceneContextMenuEvent *event) {
    if (g_mainWindow) {
        // Since event->screenPos() returns a QPoint, no toPoint() call is necessary.
        g_mainWindow->handleNodeContextMenu(this, event->screenPos());
    }
    event->accept();
}

#include "main.moc"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    MainWindow mainWindow;
    mainWindow.resize(1024, 768);
    mainWindow.show();
    return app.exec();
}
