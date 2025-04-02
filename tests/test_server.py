import pytest
from chroma_mcp.server import get_chroma_client, create_parser, mcp
import chromadb
import sys
import os
from unittest.mock import patch, MagicMock
import argparse
from mcp.server.fastmcp.exceptions import ToolError # Import ToolError
import json # Import json for parsing results
import pytest_asyncio # Import pytest_asyncio for async fixtures


# Add pytest-asyncio marker
pytest_plugins = ['pytest_asyncio']

@pytest.fixture(autouse=True)
def setup_test_args():
    # Modify sys.argv to provide the required arguments for all tests
    original_argv = sys.argv.copy()
    sys.argv = ['chroma-mcp', '--client-type', 'ephemeral']
    yield
    sys.argv = original_argv

@pytest.fixture
def mock_env_vars():
    """Fixture to mock environment variables and clean them up after tests."""
    original_environ = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_environ)

def test_get_chroma_client_ephemeral():
    # Test ephemeral client creation
    client = get_chroma_client()
    assert isinstance(client, chromadb.ClientAPI)

@pytest.mark.asyncio
async def test_list_collections():
    # Test list_collections tool
    result = await mcp.call_tool("chroma_list_collections", {"limit": None, "offset": None})
    assert isinstance(result, list)

@pytest.mark.asyncio
async def test_create_and_delete_collection():
    # Test collection creation and deletion
    collection_name = "test_collection"
    
    # Create collection
    create_result = await mcp.call_tool("chroma_create_collection", {"collection_name": collection_name})
    assert len(create_result) == 1  # Should return a list with one TextContent
    assert "Successfully created collection" in create_result[0].text
    
    # Delete collection
    delete_result = await mcp.call_tool("chroma_delete_collection", {"collection_name": collection_name})
    assert len(delete_result) == 1  # Should return a list with one TextContent
    assert "Successfully deleted collection" in delete_result[0].text

# New tests for argument parsing

def test_create_parser_defaults():
    """Test that the parser creates default values correctly."""
    parser = create_parser()
    args = parser.parse_args(['--client-type', 'ephemeral'])
    
    # Check default values
    assert args.client_type == 'ephemeral'
    assert args.ssl is True  # Default should be True
    assert args.dotenv_path == '.chroma_env'

def test_create_parser_all_args():
    """Test that the parser handles all arguments correctly."""
    parser = create_parser()
    args = parser.parse_args([
        '--client-type', 'http',
        '--host', 'test-host',
        '--port', '8080',
        '--ssl', 'false',
        '--dotenv-path', 'custom.env'
    ])
    
    # Check parsed values
    assert args.client_type == 'http'
    assert args.host == 'test-host'
    assert args.port == '8080'
    assert args.ssl is False
    assert args.dotenv_path == 'custom.env'

def test_create_parser_boolean_args():
    """Test that boolean arguments are parsed correctly with different formats."""
    parser = create_parser()
    
    # Test various true values
    for true_val in ['true', 'yes', '1', 't', 'y', 'True', 'YES']:
        args = parser.parse_args(['--client-type', 'ephemeral', '--ssl', true_val])
        assert args.ssl is True, f"Failed for value: {true_val}"
    
    # Test various false values
    for false_val in ['false', 'no', '0', 'f', 'n', 'False', 'NO']:
        args = parser.parse_args(['--client-type', 'ephemeral', '--ssl', false_val])
        assert args.ssl is False, f"Failed for value: {false_val}"

@patch.dict(os.environ, {
    'CHROMA_CLIENT_TYPE': 'http',
    'CHROMA_HOST': 'env-host',
    'CHROMA_PORT': '9090',
    'CHROMA_SSL': 'false'
})
def test_env_vars_override_defaults():
    """Test that environment variables override default values."""
    parser = create_parser()
    args = parser.parse_args([])  # No command line args
    
    # Environment variables should be used
    assert args.client_type == 'http'
    assert args.host == 'env-host'
    assert args.port == '9090'
    assert args.ssl is False

def test_cmd_args_override_env_vars(mock_env_vars):
    """Test that command line arguments override environment variables."""
    # Set environment variables
    os.environ['CHROMA_CLIENT_TYPE'] = 'http'
    os.environ['CHROMA_HOST'] = 'env-host'
    os.environ['CHROMA_SSL'] = 'false'
    
    parser = create_parser()
    # Override with command line args
    args = parser.parse_args([
        '--client-type', 'persistent',
        '--data-dir', '/test/dir',
        '--ssl', 'true'
    ])
    
    # Command line args should take precedence
    assert args.client_type == 'persistent'
    assert args.data_dir == '/test/dir'
    assert args.ssl is True
    # But other env vars should still be used
    assert args.host == 'env-host'

@patch('chroma_mcp.server._chroma_client', None)  # Reset the global client
@patch('chromadb.HttpClient')
def test_http_client_creation(mock_http_client, mock_env_vars):
    """Test HTTP client creation with various arguments."""
    mock_instance = MagicMock()
    mock_http_client.return_value = mock_instance
    
    # Set up command line args
    sys.argv = ['chroma-mcp', 
                '--client-type', 'http',
                '--host', 'test-host',
                '--port', '8080',
                '--ssl', 'false']
    
    client = get_chroma_client()
    
    # Check that HttpClient was called with correct args
    mock_http_client.assert_called_once()
    call_kwargs = mock_http_client.call_args.kwargs
    assert call_kwargs['host'] == 'test-host'
    assert call_kwargs['port'] == '8080'
    assert call_kwargs['ssl'] is False

@patch('chroma_mcp.server._chroma_client', None)  # Reset the global client
@patch('chromadb.HttpClient')
def test_cloud_client_creation(mock_http_client, mock_env_vars):
    """Test cloud client creation with various arguments."""
    mock_instance = MagicMock()
    mock_http_client.return_value = mock_instance
    
    # Set up command line args
    sys.argv = ['chroma-mcp', 
                '--client-type', 'cloud',
                '--tenant', 'test-tenant',
                '--database', 'test-db',
                '--api-key', 'test-api-key']
    
    client = get_chroma_client()
    
    # Check that HttpClient was called with correct args
    mock_http_client.assert_called_once()
    call_kwargs = mock_http_client.call_args.kwargs
    assert call_kwargs['host'] == 'api.trychroma.com'
    assert call_kwargs['ssl'] is True  # Always true for cloud
    assert call_kwargs['tenant'] == 'test-tenant'
    assert call_kwargs['database'] == 'test-db'
    assert call_kwargs['headers'] == {'x-chroma-token': 'test-api-key'}

@patch('chroma_mcp.server._chroma_client', None)  # Reset the global client
@patch('chromadb.PersistentClient')
def test_persistent_client_creation(mock_persistent_client, mock_env_vars):
    """Test persistent client creation."""
    mock_instance = MagicMock()
    mock_persistent_client.return_value = mock_instance
    
    # Set up command line args
    sys.argv = ['chroma-mcp', 
                '--client-type', 'persistent',
                '--data-dir', '/test/data/dir']
    
    client = get_chroma_client()
    
    # Check that PersistentClient was called with correct args
    mock_persistent_client.assert_called_once_with(path='/test/data/dir')

@patch('chroma_mcp.server._chroma_client', None)  # Reset the global client
@patch('chromadb.EphemeralClient')
def test_ephemeral_client_creation(mock_ephemeral_client, mock_env_vars):
    """Test ephemeral client creation."""
    mock_instance = MagicMock()
    mock_ephemeral_client.return_value = mock_instance
    
    # Set up command line args
    sys.argv = ['chroma-mcp', '--client-type', 'ephemeral']
    
    client = get_chroma_client()
    
    # Check that EphemeralClient was called
    mock_ephemeral_client.assert_called_once()

def test_client_type_validation():
    """Test validation of client type argument."""
    parser = create_parser()
    
    # Valid client types
    for valid_type in ['http', 'cloud', 'persistent', 'ephemeral']:
        args = parser.parse_args(['--client-type', valid_type])
        assert args.client_type == valid_type
    
    # Invalid client type
    with pytest.raises(SystemExit):
        parser.parse_args(['--client-type', 'invalid'])

def test_required_args_for_http_client():
    """Test that required arguments are enforced for HTTP client."""
    with patch('argparse.ArgumentParser.error') as mock_error:
        from chroma_mcp.server import main
        
        # Set up command line args without required host
        sys.argv = ['chroma-mcp', '--client-type', 'http']
        
        try:
            main()
        except:
            pass
        
        # Check that error was called for missing host
        mock_error.assert_called_with(
            "Host must be provided via --host flag or CHROMA_HOST environment variable when using HTTP client"
        )

def test_required_args_for_cloud_client():
    """Test that required arguments are enforced for cloud client."""
    with patch('argparse.ArgumentParser.error') as mock_error:
        from chroma_mcp.server import main
        
        # Set up command line args without required tenant/database/api-key
        sys.argv = ['chroma-mcp', '--client-type', 'cloud']
        
        try:
            main()
        except:
            pass
        
        # Check that error was called for missing api-key (the first check in the code)
        mock_error.assert_called_with(
            "API key must be provided via --api-key flag or CHROMA_API_KEY environment variable when using cloud client"
        )

# --- Tests for chroma_update_documents ---

@pytest.mark.asyncio
async def test_update_documents_success():
    """Test successful document update."""
    collection_name = "test_update_collection_success"
    doc_ids = ["doc1", "doc2"]
    initial_docs = ["Initial doc 1", "Initial doc 2"]
    initial_metadatas = [{"source": "initial"}, {"source": "initial"}]
    updated_docs = ["Updated doc 1", initial_docs[1]] # Update only first doc content
    updated_metadatas = [initial_metadatas[0], {"source": "updated"}] # Update only second doc metadata

    try:
        # 1. Create collection
        await mcp.call_tool("chroma_create_collection", {"collection_name": collection_name})

        # 2. Add initial documents
        await mcp.call_tool("chroma_add_documents", {
            "collection_name": collection_name,
            "documents": initial_docs,
            "metadatas": initial_metadatas,
            "ids": doc_ids
        })

        # 3. Update documents (pass both documents and metadatas)
        update_result = await mcp.call_tool("chroma_update_documents", {
            "collection_name": collection_name,
            "ids": doc_ids,
            "documents": updated_docs,
            "metadatas": updated_metadatas
        })
        assert len(update_result) == 1
        # Updated success message check
        assert (
            f"Successfully processed update request for {len(doc_ids)} documents"
            in update_result[0].text
        )

        # 4. Verify updates
        get_result_raw = await mcp.call_tool("chroma_get_documents", {
            "collection_name": collection_name,
            "ids": doc_ids,
            "include": ["documents", "metadatas"]
        })
        # Corrected: Parse the JSON string from TextContent
        assert len(get_result_raw) == 1
        get_result = json.loads(get_result_raw[0].text)
        assert isinstance(get_result, dict)

        assert get_result.get("ids") == doc_ids
        # Check updated document content
        assert get_result.get("documents") == updated_docs
        # Check updated metadata
        assert get_result.get("metadatas") == updated_metadatas

    finally:
        # Clean up
        await mcp.call_tool("chroma_delete_collection", {"collection_name": collection_name})

@pytest.mark.asyncio
async def test_update_documents_invalid_args():
    """Test update documents with invalid arguments."""
    collection_name = "test_update_collection_invalid"

    try:
        await mcp.call_tool("chroma_create_collection", {"collection_name": collection_name})
        await mcp.call_tool("chroma_add_documents", {
            "collection_name": collection_name,
            "documents": ["Test doc"],
            "ids": ["doc1"]
        })

        # Test with empty IDs list - Expect ToolError wrapping ValueError
        with pytest.raises(ToolError, match="The 'ids' list cannot be empty."):
            await mcp.call_tool("chroma_update_documents", {
                "collection_name": collection_name,
                "ids": [],
                "documents": ["New content"]
            })

        # Test with no update fields provided - Expect ToolError wrapping ValueError
        with pytest.raises(
            ToolError,
            match="At least one of 'embeddings', 'metadatas', or 'documents' must be provided"
        ):
            await mcp.call_tool("chroma_update_documents", {
                "collection_name": collection_name,
                "ids": ["doc1"]
                # No embeddings, metadatas, or documents
            })

    finally:
        # Clean up
        await mcp.call_tool("chroma_delete_collection", {"collection_name": collection_name})

@pytest.mark.asyncio
async def test_update_documents_collection_not_found():
    """Test updating documents in a non-existent collection."""
    # Expect ToolError wrapping the Exception from the function
    with pytest.raises(ToolError, match="Failed to get collection"):
        await mcp.call_tool("chroma_update_documents", {
            "collection_name": "non_existent_collection",
            "ids": ["doc1"],
            "documents": ["New content"]
        })

@pytest.mark.asyncio
async def test_update_documents_id_not_found():
    """Test updating a document with an ID that does not exist. Expect no exception."""
    collection_name = "test_update_id_not_found"
    try:
        await mcp.call_tool("chroma_create_collection", {"collection_name": collection_name})
        await mcp.call_tool("chroma_add_documents", {
            "collection_name": collection_name,
            "documents": ["Test doc"],
            "ids": ["existing_id"]
        })

        # Attempt to update a non-existent ID - should not raise Exception
        update_result = await mcp.call_tool("chroma_update_documents", {
            "collection_name": collection_name,
            "ids": ["non_existent_id"],
            "documents": ["New content"]
        })
        # Check the success message (even though the ID didn't exist)
        assert len(update_result) == 1
        assert "Successfully processed update request" in update_result[0].text

        # Optionally, verify that the existing document was not changed
        get_result_raw = await mcp.call_tool("chroma_get_documents", {
            "collection_name": collection_name,
            "ids": ["existing_id"],
            "include": ["documents"]
        })
        # Corrected assertion: Parse JSON and check structure/content
        assert len(get_result_raw) == 1
        get_result = json.loads(get_result_raw[0].text)
        assert isinstance(get_result, dict)
        assert "documents" in get_result
        assert isinstance(get_result["documents"], list)
        assert get_result["documents"] == ["Test doc"]

    finally:
        # Clean up
        await mcp.call_tool("chroma_delete_collection", {"collection_name": collection_name})

# --- Tests for chroma_delete_documents ---

@pytest_asyncio.fixture
async def setup_delete_test_collection():
    """Fixture to set up a collection with documents for deletion tests."""
    collection_name = "test_delete_docs_collection"
    client = get_chroma_client()

    # Ensure clean state
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    # Create collection and add documents using the client directly
    collection = client.get_or_create_collection(collection_name)
    collection.add(
        documents=["doc1 text", "doc2 text", "another doc", "doc4 special"],
        metadatas=[{"type": "a", "val": 1}, {"type": "b", "val": 2}, {"type": "a", "val": 3}, {"type": "c", "val": 4}],
        ids=["id1", "id2", "id3", "id4"]
    )

    yield collection_name

    # Teardown: Delete the collection after the test
    try:
        client.delete_collection(collection_name)
    except Exception as e:
        print(f"Error during teardown deleting collection {collection_name}: {e}")


@pytest.mark.asyncio
async def test_delete_documents_by_ids(setup_delete_test_collection):
    """Test deleting documents by providing a list of IDs."""
    collection_name = setup_delete_test_collection
    ids_to_delete = ["id1", "id3"]

    delete_result = await mcp.call_tool("chroma_delete_documents", {
        "collection_name": collection_name,
        "ids": ids_to_delete
    })
    # Check result type and content (assuming tool returns string in a list)
    assert isinstance(delete_result, list)
    assert len(delete_result) >= 1
    assert hasattr(delete_result[0], 'text')
    assert f"Successfully processed delete request for collection '{collection_name}'" in delete_result[0].text

    # Verify deletion using the client directly
    client = get_chroma_client()
    collection = client.get_collection(collection_name)
    remaining_docs = collection.get(include=["metadatas"])
    assert "id1" not in remaining_docs["ids"]
    assert "id3" not in remaining_docs["ids"]
    assert "id2" in remaining_docs["ids"]
    assert "id4" in remaining_docs["ids"]
    assert len(remaining_docs["ids"]) == 2

@pytest.mark.asyncio
async def test_delete_documents_by_where(setup_delete_test_collection):
    """Test deleting documents using a 'where' filter."""
    collection_name = setup_delete_test_collection
    where_filter = {"type": "a"}

    delete_result = await mcp.call_tool("chroma_delete_documents", {
        "collection_name": collection_name,
        "where": where_filter
    })
    assert isinstance(delete_result, list)
    assert len(delete_result) >= 1
    assert hasattr(delete_result[0], 'text')
    assert f"Successfully processed delete request for collection '{collection_name}'" in delete_result[0].text

    # Verify deletion
    client = get_chroma_client()
    collection = client.get_collection(collection_name)
    remaining_docs = collection.get(include=["metadatas"])
    assert "id1" not in remaining_docs["ids"]
    assert "id3" not in remaining_docs["ids"]
    assert "id2" in remaining_docs["ids"]
    assert "id4" in remaining_docs["ids"]
    assert len(remaining_docs["ids"]) == 2
    for meta in remaining_docs["metadatas"]:
        assert meta["type"] != "a"

@pytest.mark.asyncio
async def test_delete_documents_by_where_document(setup_delete_test_collection):
    """Test deleting documents using a 'where_document' filter."""
    collection_name = setup_delete_test_collection
    where_doc_filter = {"$contains": "special"}

    delete_result = await mcp.call_tool("chroma_delete_documents", {
        "collection_name": collection_name,
        "where_document": where_doc_filter
    })
    assert isinstance(delete_result, list)
    assert len(delete_result) >= 1
    assert hasattr(delete_result[0], 'text')
    assert f"Successfully processed delete request for collection '{collection_name}'" in delete_result[0].text

    # Verify deletion
    client = get_chroma_client()
    collection = client.get_collection(collection_name)
    remaining_docs = collection.get(include=["documents"])
    assert "id4" not in remaining_docs["ids"]
    assert "id1" in remaining_docs["ids"]
    assert "id2" in remaining_docs["ids"]
    assert "id3" in remaining_docs["ids"]
    assert len(remaining_docs["ids"]) == 3
    for doc in remaining_docs["documents"]:
        assert "special" not in doc


@pytest.mark.asyncio
async def test_delete_documents_no_criteria_error(setup_delete_test_collection):
    """Test error when no deletion criteria are provided."""
    collection_name = setup_delete_test_collection

    with pytest.raises(ToolError, match="No deletion criteria provided"):
        await mcp.call_tool("chroma_delete_documents", {
            "collection_name": collection_name
        })

@pytest.mark.asyncio
async def test_delete_documents_both_ids_and_filter_error(setup_delete_test_collection):
    """Test error when both 'ids' and a filter are provided."""
    collection_name = setup_delete_test_collection

    with pytest.raises(ToolError, match="Cannot provide both 'ids' and filtering conditions"):
        await mcp.call_tool("chroma_delete_documents", {
            "collection_name": collection_name,
            "ids": ["id1"],
            "where": {"type": "a"}
        })

    with pytest.raises(ToolError, match="Cannot provide both 'ids' and filtering conditions"):
        await mcp.call_tool("chroma_delete_documents", {
            "collection_name": collection_name,
            "ids": ["id1"],
            "where_document": {"$contains": "text"}
        })

@pytest.mark.asyncio
async def test_delete_documents_nonexistent_collection():
    """Test error when trying to delete from a non-existent collection."""
    collection_name = "nonexistent_collection_for_delete"

    with pytest.raises(ToolError): # Catch ToolError wrapping the underlying exception
         await mcp.call_tool("chroma_delete_documents", {
            "collection_name": collection_name,
            "ids": ["id_does_not_matter"]
        })

@pytest.mark.asyncio
async def test_delete_documents_nonexistent_ids(setup_delete_test_collection):
    """Test deleting non-existent IDs does not raise an error."""
    collection_name = setup_delete_test_collection
    ids_to_delete = ["nonexistent_id1", "nonexistent_id2"]

    delete_result = await mcp.call_tool("chroma_delete_documents", {
        "collection_name": collection_name,
        "ids": ids_to_delete
    })
    assert isinstance(delete_result, list)
    assert len(delete_result) >= 1
    assert hasattr(delete_result[0], 'text')
    assert f"Successfully processed delete request for collection '{collection_name}'" in delete_result[0].text

    # Verify no documents were actually deleted
    client = get_chroma_client()
    collection = client.get_collection(collection_name)
    count_after = collection.count()
    assert count_after == 4